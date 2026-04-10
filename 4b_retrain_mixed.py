# 4b_rl_retrain_mixed.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Re-train the RL policy using:
#     • 70 % experiences from your original dataset (attack + normal)
#     • 30 % experiences from your real hardware normal traffic
#
#   This teaches the RL agent that hardware-style normal traffic should
#   always get action=ALLOW, so it stops false-alarming on your sensors.
#
# WHAT CHANGES vs rl_train.py:
#   • Uses the fine-tuned detector (_ft) and scaler (_ft) from step 4a.
#   • Adds a HardwareNormalBuffer that replays label=0 sequences from
#     your real capture inside the DQN experience replay.
#   • Saves new RL files with suffix _ft so originals are untouched.
#   • Everything else (QNet, Replay, DQN loop, CSV logging) is identical.
#
# RUN:
#   python 4b_rl_retrain_mixed.py [--detector cnn_only|cnn_attn|cnn_bilstm_attn|all]
# ─────────────────────────────────────────────────────────────────────────────

import random
import csv
import math
import pickle
from pathlib import Path
from collections import Counter
import argparse

import numpy as np
import torch
from torch import nn
from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP

import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler

# ── CONSTANTS (copied from rl_env.py — do NOT change) ────────────────────────
#these action IDs must exactly match rl_env.py
#if they are different the reward function would punish the wrong actions

A_ALLOW          = 0
A_RATE_LIMIT     = 1
A_TEMP_BLOCK     = 2
A_PERM_BLOCK     = 3
A_DROP_SYN       = 4
A_DROP_CONNECT   = 5
A_DELAY_CONNECT  = 6
A_LIMIT_PUBLISH  = 7
A_BLOCK_SUBSCRIBE= 8
A_DISCONNECT     = 9
A_QUARANTINE     = 10
A_ISOLATE_NODE   = 11
A_REDUCE_QOS     = 12
A_ALERT_ONLY     = 13
A_ESCALATE       = 14
A_DEESCALATE     = 15

ACTION_NAMES = {
    0: "ALLOW", 1: "RATE_LIMIT_IP", 2: "TEMP_BLOCK_IP", 3: "PERM_BLOCK_IP",
    4: "DROP_SYN_DELAY_TCP", 5: "DROP_CONNECT", 6: "DELAY_CONNECT",
    7: "LIMIT_PUBLISH", 8: "BLOCK_SUBSCRIBE", 9: "DISCONNECT_CLIENT",
    10: "QUARANTINE_CLIENT", 11: "ISOLATE_NODE", 12: "REDUCE_QOS",
    13: "ALERT_ONLY", 14: "ESCALATE", 15: "DEESCALATE",
}
#16 possible actions the RL agent can take
#ALLOW means traffic is safe and passes through
#TEMP_BLOCK/PERM_BLOCK/QUARANTINE are heavy-handed actions for confirmed attacks
#ALERT_ONLY and DEESCALATE are safe soft actions for uncertain situations

SAFE_ACTIONS         = {A_ALLOW, A_ALERT_ONLY, A_DEESCALATE}
#safe actions cause no harm to legitimate users — we prefer these when unsure
HEAVY_ACTIONS        = {A_TEMP_BLOCK, A_PERM_BLOCK, A_DISCONNECT, A_QUARANTINE, A_ISOLATE_NODE}
#heavy actions strongly block or isolate — only justified for confirmed attacks
NON_MITIGATE_ACTIONS = {A_ALLOW, A_ALERT_ONLY, A_ESCALATE, A_DEESCALATE}
#non-mitigate means the agent did not actually stop the attack
#if attack traffic gets a non-mitigate action it counts as a miss (FN)


# ── FEATURE COLUMNS (mirrors rl_env.py add_features output) ──────────────────

FEATURE_COLS = [
    "Time", "time_delta", "Length",
    "has_mqtt_port",
    "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
    "to_mqtt", "from_mqtt",
]
#the same 12 features used in training — order matters because models index by position
_MQTT_PORTS = {1883, 8883}


def is_tcp_mqtt_port_df(df: pd.DataFrame) -> pd.Series:
    proto = df.get("Protocol", pd.Series([""] * len(df))).astype(str).str.upper()
    info = df.get("Info", pd.Series([""] * len(df))).astype(str).str.upper()
    mask_tcp_only = (proto == "TCP")
    mask_mqtt_ports = info.str.contains(r"\b1883\b|\b8883\b", regex=True, na=False)
    return mask_tcp_only & mask_mqtt_ports
    #same filter as step 1 and step 4a to make sure all data goes through identical preprocessing


def _add_features_one_file(g: pd.DataFrame) -> np.ndarray:
    """
    Applies the same feature engineering as rl_env.add_features but for a
    single file group (a small DataFrame), so no large index sort is needed.
    Returns a float32 numpy array with columns matching FEATURE_COLS.
    """
    g = g.copy()
    g["Time"]   = pd.to_numeric(g["Time"],   errors="coerce").fillna(0.0)
    g["Length"] = pd.to_numeric(g["Length"], errors="coerce").fillna(0.0)

    # relative time within this file — no cross-file groupby needed
    g["Time"] = g["Time"] - g["Time"].min()
    #make time start from 0 within this file
    #so sequences from different files are on the same time scale
    g["time_delta"] = g["Time"].diff().fillna(0.0)
    #time gap between consecutive packets within the same file

    info  = g["Info"].astype(str).str.upper()
    ports = info.str.extract(r"(\d+)\s*→\s*(\d+)")
    sport = pd.to_numeric(ports[0], errors="coerce")
    dport = pd.to_numeric(ports[1], errors="coerce")

    g["has_mqtt_port"] = ((sport.isin(_MQTT_PORTS)) | (dport.isin(_MQTT_PORTS))).fillna(False).astype(int)
    g["to_mqtt"]       = (dport.isin(_MQTT_PORTS)).fillna(False).astype(int)
    g["from_mqtt"]     = (sport.isin(_MQTT_PORTS)).fillna(False).astype(int)
    g["flag_syn"]      = info.str.contains(r"\bSYN\b", regex=True, na=False).astype(int)
    g["flag_ack"]      = info.str.contains(r"\bACK\b", regex=True, na=False).astype(int)
    g["flag_fin"]      = info.str.contains(r"\bFIN\b", regex=True, na=False).astype(int)
    g["flag_rst"]      = info.str.contains(r"\bRST\b", regex=True, na=False).astype(int)
    g["flag_psh"]      = info.str.contains(r"\bPSH\b", regex=True, na=False).astype(int)
    g["flag_urg"]      = info.str.contains(r"\bURG\b", regex=True, na=False).astype(int)
    #all 12 features computed from raw Wireshark CSV columns
    #each row becomes one packet's feature vector

    return g[FEATURE_COLS].to_numpy(dtype=np.float32)
# ── MEMORY-SAFE SEQUENCE BUILDERS (replace your current _df_to_sequences and build_rl_data_from_csv) ──

def _iter_file_groups_from_split_csv(split_csv_path: Path, needed_cols: list[str], chunk_size: int = 200_000):
    """
    Stream one split CSV and yield one whole file at a time, without loading
    the full split in RAM and without using pandas groupby.

    This relies on the fact that rows from the same original capture file stay
    together in the labeled dataset and in the saved row-order split CSVs.
    """
    if not split_csv_path.exists():
        return
    #if the split CSV does not exist we just stop — no error needed

    carry_parts = []
    carry_name = None
    #carry is used to handle files that get split across two chunks
    #when a file's rows don't all fit in one chunk we save the partial piece
    #and combine it with the next chunk's rows for the same file

    for chunk in pd.read_csv(
        split_csv_path,
        usecols=needed_cols,
        dtype={"label": "int8"},
        chunksize=chunk_size,
    ):
        #read the CSV in small pieces (chunks) instead of all at once
        #this keeps memory usage low even for very large datasets
        if len(chunk) == 0:
            continue

        file_names = chunk["file_name"].astype(str)

        start = 0
        if carry_name is not None:
            same_prefix = 0
            while same_prefix < len(chunk) and file_names.iloc[same_prefix] == carry_name:
                same_prefix += 1
            #count how many rows at the start of this chunk belong to the carried file

            if same_prefix > 0:
                carry_parts.append(chunk.iloc[:same_prefix].copy())
                start = same_prefix

            if start < len(chunk) and carry_parts:
                yield carry_name, pd.concat(carry_parts, ignore_index=True)
                #yield the complete carried file now that we have all its rows
                carry_parts = []
                carry_name = None

        while start < len(chunk):
            cur_name = file_names.iloc[start]
            end = start + 1
            while end < len(chunk) and file_names.iloc[end] == cur_name:
                end += 1
            #find where this file's rows end in the current chunk

            part = chunk.iloc[start:end].copy()

            if end == len(chunk):
                #the file might continue in the next chunk so we carry it
                carry_name = cur_name
                carry_parts = [part]
            else:
                yield cur_name, part
                #the full file is within this chunk so yield it now

            start = end

    if carry_parts:
        yield carry_name, pd.concat(carry_parts, ignore_index=True)
        #yield whatever was left being carried after the last chunk


def _count_sequences_in_split_csv(
        split_csv_path: Path,
        seq_len: int,
        step: int,
        needed_cols: list[str],
        chunk_size: int = 200_000,
) -> int:
    total_seq = 0
    for _, g in _iter_file_groups_from_split_csv(split_csv_path, needed_cols, chunk_size=chunk_size):
        n = len(g)
        if n >= seq_len:
            total_seq += 1 + (n - seq_len) // step
    return int(total_seq)
    #count how many sequences we can build from this split without actually building them
    #we need this count first so we can pre-allocate the right amount of disk space


def _split_csv_to_sequences(
        split_csv_path: Path,
        seq_len: int,
        step: int,
        split_name: str,
        cache_dir: Path,
        needed_cols: list[str],
        chunk_size: int = 200_000,
) -> tuple:
    """
    Builds sequences directly from one temporary split CSV in a streaming way.
    This avoids both large concat() calls and large groupby() allocations.
    """
    if not split_csv_path.exists():
        raise RuntimeError(f"No rows found for split '{split_name}'.")

    total_seq = _count_sequences_in_split_csv(
        split_csv_path, seq_len, step, needed_cols, chunk_size=chunk_size
    )
    #first count how many sequences we will build so we can allocate disk space

    if total_seq == 0:
        raise RuntimeError(f"No sequences created for split '{split_name}' — reduce SEQ_LEN or check data.")

    print(f"[INFO] {split_name}: allocating disk-backed arrays for {total_seq:,} sequences...")

    cache_dir.mkdir(parents=True, exist_ok=True)

    x_path = cache_dir / f"{split_name}_X.dat"
    y_path = cache_dir / f"{split_name}_y.dat"
    f_path = cache_dir / f"{split_name}_files.dat"
    #we save to disk-backed memory-mapped files instead of RAM
    #this is called a memmap and it lets us work with huge datasets
    #that would never fit entirely in RAM

    for p in [x_path, y_path, f_path]:
        if p.exists():
            p.unlink()
    #delete old cache files from previous runs to start fresh

    X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(total_seq, seq_len, len(FEATURE_COLS)))
    y_mm = np.memmap(y_path, dtype=np.int64, mode="w+", shape=(total_seq,))
    f_mm = np.memmap(f_path, dtype=np.int32, mode="w+", shape=(total_seq,))
    #create memory-mapped files on disk with the exact needed shape
    #mode="w+" means create and allow writing

    pos = 0
    file_code_map: Dict[str, int] = {}
    next_file_code = 0
    #we assign each unique file name a numeric code because strings can't go in numpy arrays

    for file_name, g in _iter_file_groups_from_split_csv(split_csv_path, needed_cols, chunk_size=chunk_size):
        g = g.sort_values("Time")
        #sort by time within each file to make sure packets are in the right order
        X = _add_features_one_file(g)
        y = g["label"].astype(int).to_numpy()

        if len(X) < seq_len:
            continue
        #skip files that are too short to form even one sequence

        if file_name not in file_code_map:
            file_code_map[file_name] = next_file_code
            next_file_code += 1
        file_code = file_code_map[file_name]

        for start in range(0, len(X) - seq_len + 1, step):
            end = start + seq_len
            X_mm[pos] = X[start:end]
            y_mm[pos] = int(y[start:end].max())
            #if any packet in the window is an attack we label the whole sequence as attack
            #this is conservative: one attack packet in 20 is enough to flag the window
            f_mm[pos] = int(file_code)
            pos += 1

    X_mm.flush()
    y_mm.flush()
    f_mm.flush()
    #flush writes all pending data from memory to disk so nothing is lost

    X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=(total_seq, seq_len, len(FEATURE_COLS)))
    y_mm = np.memmap(y_path, dtype=np.int64, mode="r", shape=(total_seq,))
    f_mm = np.memmap(f_path, dtype=np.int32, mode="r", shape=(total_seq,))
    #reopen in read-only mode — from now on we only read not write

    return X_mm, y_mm, f_mm


def build_rl_data_from_csv(
        csv_all: Path,
        train_idx: Path,
        val_idx: Path,
        test_idx: Path,
        seq_len: int = 20,
        step: int = 5,
        chunk_size: int = 200_000,
) -> Dict[str, Dict[str, np.ndarray]]:
    tr = np.load(train_idx)
    va = np.load(val_idx)
    te = np.load(test_idx)
    #load the row index arrays saved by prepare_split.py
    #these tell us which rows belong to train, val and test

    # only the needed columns
    NEEDED_COLS = ["Time", "Length", "Info", "label", "file_name"]

    print("[INFO] Reading CSV in chunks (memory-efficient mode)...")

    cache_dir = OUT_DIR / "rl_memmap_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_csvs = {
        "train": cache_dir / "train_split_rows.csv",
        "val":   cache_dir / "val_split_rows.csv",
        "test":  cache_dir / "test_split_rows.csv",
    }
    #we will write temporary CSVs for each split
    #this avoids loading the entire labeled dataset into RAM at once

    for p in split_csvs.values():
        if p.exists():
            p.unlink()
    #delete old split CSVs to avoid mixing data from previous runs

    wrote_header = {"train": False, "val": False, "test": False}

    global_row = 0
    for chunk in pd.read_csv(
        csv_all,
        usecols=NEEDED_COLS,
        dtype={"label": "int8"},
        chunksize=chunk_size,
    ):
        #read the big labeled CSV in chunks so we never load it all into RAM
        chunk_len = len(chunk)
        chunk_idx = np.arange(global_row, global_row + chunk_len)
        global_row += chunk_len

        in_tr = np.isin(chunk_idx, tr)
        in_va = np.isin(chunk_idx, va)
        in_te = np.isin(chunk_idx, te)
        #check which rows in this chunk belong to which split

        if in_tr.any():
            part = chunk.loc[in_tr].copy()
            part.to_csv(split_csvs["train"], mode="a", header=not wrote_header["train"], index=False)
            wrote_header["train"] = True
            del part
        if in_va.any():
            part = chunk.loc[in_va].copy()
            part.to_csv(split_csvs["val"], mode="a", header=not wrote_header["val"], index=False)
            wrote_header["val"] = True
            del part
        if in_te.any():
            part = chunk.loc[in_te].copy()
            part.to_csv(split_csvs["test"], mode="a", header=not wrote_header["test"], index=False)
            wrote_header["test"] = True
            del part
        #write each split's rows to their own temporary CSV file
        #mode="a" means append so we keep adding chunks without overwriting

    print("[INFO] Assembling splits and building sequences per file...")

    Xtr, ytr, ftr = _split_csv_to_sequences(
        split_csvs["train"], seq_len, step, "train", cache_dir, NEEDED_COLS, chunk_size=chunk_size
    )
    Xva, yva, fva = _split_csv_to_sequences(
        split_csvs["val"], seq_len, step, "val", cache_dir, NEEDED_COLS, chunk_size=chunk_size
    )
    Xte, yte, fte = _split_csv_to_sequences(
        split_csvs["test"], seq_len, step, "test", cache_dir, NEEDED_COLS, chunk_size=chunk_size
    )
    #build disk-backed sequence arrays for each split
    #each element is a (20, 12) matrix representing one sliding window of 20 packets

    print(f"[INFO] Sequences — train: {len(Xtr):,}  val: {len(Xva):,}  test: {len(Xte):,}")
    return {
        "train": {"X": Xtr, "y": ytr, "files": ftr},
        "val":   {"X": Xva, "y": yva, "files": fva},
        "test":  {"X": Xte, "y": yte, "files": fte},
        "feature_cols": {"cols": np.array(FEATURE_COLS, dtype=object)},
    }

# ── NeuroGuardRLEnv (copied from rl_env.py — do NOT change) ──────────────────
#the environment class is duplicated here so 4b is self-contained
#it must be identical to rl_env.py or the reward signals will be wrong

class _MultiHeadAttnEnv(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.scale     = math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores       = (Q @ K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        out          = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out).mean(dim=1)


class _CNN_Only_Env(nn.Module):
    def __init__(self, feat_dim, seq_len=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * seq_len, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.fc(self.conv(x.transpose(1, 2))).squeeze(1)


class _CNN_Attention_Env(nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.multi_head_attn = _MultiHeadAttnEnv(128, num_heads)
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.fc(self.multi_head_attn(h)).squeeze(1)


class _CNN_BiLSTM_Attn_Env(nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.multi_head_attn = _MultiHeadAttnEnv(128, num_heads)
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        h, _ = self.bilstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        return self.fc(self.multi_head_attn(h)).squeeze(1)


class NeuroGuardRLEnv:
    #this is the RL environment — it simulates the network where the agent makes decisions
    #the agent receives a state, takes an action, and the environment returns a reward
    #the reward teaches the agent which actions are good and which are harmful

    det_attack_thr   = 0.5
    #if the detector outputs probability >= 0.5 we treat the sequence as an attack
    det_normal_low_thr = 0.3
    #if the detector outputs probability < 0.3 we are confident it is normal traffic

    action_costs = {
        A_ALLOW: 0.00, A_RATE_LIMIT: 0.15, A_DROP_SYN: 0.20,
        A_DROP_CONNECT: 0.30, A_DELAY_CONNECT: 0.20, A_LIMIT_PUBLISH: 0.25,
        A_BLOCK_SUBSCRIBE: 0.25, A_REDUCE_QOS: 0.20, A_ALERT_ONLY: 0.05,
        A_ESCALATE: 0.40, A_DEESCALATE: 0.10,
        A_TEMP_BLOCK: 0.60, A_PERM_BLOCK: 1.00,
        A_DISCONNECT: 0.80, A_QUARANTINE: 1.00, A_ISOLATE_NODE: 1.20,
    }
    #each action has a cost that is subtracted from the reward
    #heavy actions like permanent block have high costs because they disrupt legitimate users
    #light actions like rate limiting have low costs because they are easily reversible

    def __init__(self, X_seq, y_seq, files_seq,
                 detector_ckpt, scaler_ckpt, detector_type,
                 device="cpu", max_steps_per_episode=200):
        self.X_seq   = X_seq
        self.y_seq   = y_seq
        self.files_seq = files_seq
        self.device  = device
        self.max_steps_per_episode = max_steps_per_episode
        self.detector_type = detector_type

        # load scaler
        with open(scaler_ckpt, "rb") as f:
            self.scaler = pickle.load(f)
        #load the fine-tuned scaler from 4a so scaling matches real hardware statistics

        # load detector
        if detector_type == "cnn_only":
            self.detector = _CNN_Only_Env(feat_dim=FEAT_DIM, seq_len=SEQ_LEN)
        elif detector_type == "cnn_attention":
            self.detector = _CNN_Attention_Env(feat_dim=FEAT_DIM, num_heads=4)
        elif detector_type == "cnn_bilstm_attn":
            self.detector = _CNN_BiLSTM_Attn_Env(feat_dim=FEAT_DIM, num_heads=4)
        self.detector.load_state_dict(torch.load(detector_ckpt, map_location=device))
        self.detector.eval()
        #load the fine-tuned detector checkpoint from 4a
        #eval() mode disables dropout so the detector gives consistent outputs

        self.file_to_indices: Dict[str, List[int]] = {}
        for i, fn in enumerate(self.files_seq):
            key = str(fn)
            self.file_to_indices.setdefault(key, []).append(i)
        self.file_names = list(self.file_to_indices.keys())
        #group sequence indices by which capture file they came from
        #during training each episode plays through all sequences from one file
        #this is realistic because a real attack comes from one continuous traffic capture

        self.cur_file: str | None   = None
        self.cur_indices: List[int] | None = None
        self.t = 0
        self.steps = 0
        self.prev_action     = A_ALLOW
        self.escalation_level= 0
        self.fp_counter      = 0
        self.fn_counter      = 0
        #state variables reset at the start of each episode
        #fp_counter tracks how many false alarms happened recently
        #fn_counter tracks how many attacks were missed recently
        #these go into the state vector so the agent knows its own recent history

    def _scale_seq(self, seq_x: np.ndarray) -> np.ndarray:
        continuous_idx = [0, 1, 2]
        seq_x_scaled = seq_x.copy()
        seq_x_scaled[:, continuous_idx] = self.scaler.transform(seq_x[:, continuous_idx])
        return seq_x_scaled
        #apply the scaler to the 3 continuous features (Time, time_delta, Length)
        #the 9 binary flag features are not scaled because they are already 0 or 1

    def _detector_prob(self, seq_x: np.ndarray) -> float:
        seq_x_scaled = self._scale_seq(seq_x)
        xb = torch.tensor(seq_x_scaled[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logit = self.detector(xb)
            p = torch.sigmoid(logit).item()
        return float(p)
        #run the sequence through the fine-tuned detector and get attack probability
        #the detector's output is the most important feature the RL agent sees

    def _make_state(self, seq_x: np.ndarray, det_p: float) -> np.ndarray:
        seq_x_scaled       = self._scale_seq(seq_x)
        time_delta_mean    = float(np.mean(seq_x_scaled[:, 1]))
        length_mean        = float(np.mean(seq_x_scaled[:, 2]))
        has_mqtt_port_mean = float(np.mean(seq_x_scaled[:, 3]))
        flags_mean         = np.mean(seq_x_scaled[:, 4:12], axis=0).astype(np.float32)
        extras = np.array([
            det_p,
            #detector probability: the most important single number — how sure are we it is an attack
            time_delta_mean,
            #average gap between packets — small means fast/flooding, large means slow/normal
            length_mean,
            #average packet size — attacks often have uniform tiny or giant packets
            has_mqtt_port_mean,
            #fraction of packets hitting MQTT ports — attacks target these specifically
            float(self.prev_action) / 15.0,
            #what action was taken last step (normalised to 0-1 range)
            float(self.escalation_level) / 5.0,
            #current escalation level (0 to 5) normalised — higher means we already escalated
            float(self.fp_counter) / 10.0,
            #false positive counter (0 to 10) normalised — high means we are over-blocking recently
            float(self.fn_counter) / 10.0,
            #false negative counter (0 to 10) normalised — high means we are missing attacks recently
        ], dtype=np.float32)
        return np.concatenate([extras, flags_mean], axis=0).astype(np.float32)
        #final state vector is 16 dimensions (8 extras + 8 flag means)
        #this is what the Q-network receives as input for every decision

    def reset(self, file_name: str | None = None) -> np.ndarray:
        if file_name is None:
            self.cur_file = str(np.random.choice(self.file_names))
        else:
            self.cur_file = str(file_name)
        self.cur_indices      = self.file_to_indices[self.cur_file]
        self.t                = 0
        self.steps            = 0
        self.prev_action      = A_ALLOW
        self.escalation_level = 0
        self.fp_counter       = 0
        self.fn_counter       = 0
        #reset all episode state and pick a random capture file to replay
        idx   = self.cur_indices[self.t]
        seq_x = self.X_seq[idx]
        det_p = self._detector_prob(seq_x)
        return self._make_state(seq_x, det_p)
        #return the initial state so the agent can take its first action

    def step(self, action: int):
        action  = int(action)
        if self.cur_indices is None:
            raise RuntimeError("Call reset() before step().")
        idx    = self.cur_indices[self.t]
        seq_x  = self.X_seq[idx]
        y_true = int(self.y_seq[idx])
        det_p  = self._detector_prob(seq_x)
        det_attack = det_p >= self.det_attack_thr
        reward = 0.0
        is_cnn_only = (self.detector_type == "cnn_only")

        if y_true == 1:
            #this sequence contains an attack
            if action == A_ALLOW:
                reward -= 2.0 if is_cnn_only else 3.0
                #letting an attack through is a serious mistake — large negative reward
                self.fn_counter = min(self.fn_counter + (1 if is_cnn_only else 2), 10)
            elif action in NON_MITIGATE_ACTIONS:
                reward -= 0.5 if is_cnn_only else 0.8
                #soft non-actions on attacks are still wrong but less bad than ALLOW
                self.fn_counter = min(self.fn_counter + 1, 10)
            else:
                reward += 1.5 if is_cnn_only else 1.2
                #any real mitigation action on an attack is rewarded
                if det_attack:
                    reward += 0.5 if is_cnn_only else 0.3
                    #bonus if the detector also agreed it was an attack — confident correct decision
                if action in HEAVY_ACTIONS:
                    reward -= 0.1 if is_cnn_only else 0.2
                    #small penalty for using a heavy action like permanent block
                    #we want proportional responses not always reaching for the biggest hammer
                self.fn_counter = max(self.fn_counter - 1, 0)
        else:
            #this sequence is normal traffic
            if action == A_ALLOW:
                reward += 2.5 if is_cnn_only else 2.0
                #correctly allowing normal traffic is the most common correct action
                self.fp_counter = max(self.fp_counter - 2, 0)
            elif action in SAFE_ACTIONS:
                reward += 0.8 if is_cnn_only else 0.6
                #soft safe actions on normal traffic are slightly rewarded
                self.fp_counter = max(self.fp_counter - 1, 0)
            else:
                reward -= 2.0 if is_cnn_only else 2.5
                #blocking or restricting normal traffic is a false alarm — heavily penalised
                self.fp_counter = min(self.fp_counter + 2, 10)
            if det_p < self.det_normal_low_thr and action != A_ALLOW:
                reward -= 1.0 if is_cnn_only else 1.5
                #extra penalty: the detector was very confident it was normal but we blocked anyway
            if action == A_ESCALATE:
                reward -= 2.0 if is_cnn_only else 2.5
                #escalating on clearly normal traffic is an overreaction — penalise it

        reward -= float(self.action_costs.get(action, 0.0))
        #subtract the action's operational cost from the reward

        if action == A_ESCALATE:
            self.escalation_level = min(self.escalation_level + 1, 5)
        elif action == A_DEESCALATE:
            self.escalation_level = max(self.escalation_level - 1, 0)
        #update the escalation level based on the chosen action
        #this feeds back into the next step's state vector

        self.prev_action = action
        self.t     += 1
        self.steps += 1

        done = (self.t >= len(self.cur_indices)) or (self.steps >= self.max_steps_per_episode)
        #episode is done when we run out of sequences in the current file
        #or when we hit the maximum number of steps per episode
        if done:
            next_state = np.zeros(16, dtype=np.float32)
        else:
            next_idx   = self.cur_indices[self.t]
            next_seq_x = self.X_seq[next_idx]
            next_p     = self._detector_prob(next_seq_x)
            next_state = self._make_state(next_seq_x, next_p)

        info = {
            "file":           self.cur_file,
            "det_p":          float(det_p),
            "y_true":         int(y_true),
            "action":         int(action),
            "action_name":    ACTION_NAMES.get(int(action), "UNKNOWN"),
            "is_safe_action": bool(action in SAFE_ACTIONS),
            "is_heavy_action":bool(action in HEAVY_ACTIONS),
        }
        return next_state, float(reward), bool(done), info

# ── PATHS ─────────────────────────────────────────────────────────────────────
OUT_DIR   = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")
PCAP_FILE = Path(r"C:\Users\User\Downloads\broker_data_transfer.csv")   # <-- real capture CSV
#the same real hardware CSV from step 4a
#we use it here to build the HardwareNormalBuffer for mixed training

CSV_ALL   = OUT_DIR / "mqtt_packets_labeled.csv"
TRAIN_IDX = OUT_DIR / "train_idx.npy"
VAL_IDX   = OUT_DIR / "val_idx.npy"
TEST_IDX  = OUT_DIR / "test_idx.npy"

# ── FINE-TUNED MODEL CONFIGS (output of 4a_transfer_learning.py) ──────────────
DETECTOR_CONFIGS = {
    "cnn_only": {
        "ckpt":   OUT_DIR / "detector_cnn_only_ft.pt",
        #we use the fine-tuned checkpoint from 4a not the original one
        "scaler": OUT_DIR / "scaler_cnn_only_ft.pkl",
        #we also use the fine-tuned scaler so scaling matches real hardware statistics
        "type":   "cnn_only",
    },
    "cnn_attn": {
        "ckpt":   OUT_DIR / "detector_cnn_attention_ft.pt",
        "scaler": OUT_DIR / "scaler_cnn_attention_ft.pkl",
        "type":   "cnn_attention",
    },
    "cnn_bilstm_attn": {
        "ckpt":   OUT_DIR / "detector_cnn_bilstm_attn_ft.pt",
        "scaler": OUT_DIR / "scaler_cnn_bilstm_attn_ft.pkl",
        "type":   "cnn_bilstm_attn",
    },
}

# ── HYPER-PARAMS (identical to rl_train.py) ───────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN        = 20
STEP           = 5
FEAT_DIM       = 12
MQTT_PORTS     = {1883, 8883}

N_ACTIONS      = 16
STATE_DIM      = 16
#state is always 16 dimensions — 8 traffic features + 8 flag means (see _make_state)

EPISODES       = 225
#we train for 225 episodes which is the same as the original RL training
MAX_STEPS_EP   = 200
#maximum 200 steps per episode to prevent one long file from dominating training

GAMMA          = 0.99
#discount factor — rewards far in the future are worth almost as much as immediate ones
#0.99 means the agent plans ahead but cares slightly more about immediate outcomes
LR             = 1e-3
#learning rate for the Q-network — controls how fast it updates each step
BATCH          = 256
#mini-batch size drawn from the replay buffer for each Q-network update
REPLAY_SIZE    = 200_000
#the replay buffer stores up to 200,000 past experiences
#when full the oldest ones are overwritten (circular buffer)
START_LEARNING = 5_000
#we wait until 5,000 experiences are collected before starting to train
#this ensures the buffer has enough variety before we start learning from it
TARGET_UPDATE  = 1_000
#every 1,000 steps we copy the main Q-network weights to the target network
#the target network provides stable Q-value targets so training does not oscillate

EPS_START      = 1.0
EPS_END        = 0.05
EPS_DECAY_STEPS= 50_000
#epsilon-greedy exploration: we start by taking random actions (epsilon=1.0)
#and slowly reduce to mostly greedy actions (epsilon=0.05) over 50,000 steps
#this ensures the agent explores enough before committing to a strategy

VAL_EVAL_EPISODES = 30
VAL_SEED          = 123
#evaluate on 30 validation episodes with a fixed random seed for reproducibility

MAX_FPR = 0.05
MAX_FNR = 0.01
MIN_TPR = 0.99
MIN_TNR = 0.95
#safety thresholds — a model is only "accepted" if it meets all four
#FPR ≤ 5% means at most 5% of normal traffic is wrongly blocked
#FNR ≤ 1% means at most 1% of attacks are missed (very strict)
#TPR ≥ 99% means at least 99% of attacks are caught
#TNR ≥ 95% means at least 95% of normal traffic is correctly allowed

# fraction of each training batch drawn from hardware normal replay
HW_BATCH_FRAC  = 0.30   # 30 % hardware, 70 % dataset
#30% of every training batch comes from the real hardware normal buffer
#70% comes from the original simulation dataset (which has both attacks and normal)
#this mixed training ensures the RL agent learns to ALLOW real hardware traffic
#without forgetting how to handle attacks from the simulation data

# reward given to RL for correctly allowing hardware normal traffic
HW_ALLOW_REWARD = +1.5   # strong positive: this is definitely normal
#when the agent replays a hardware normal experience and the action is ALLOW
#it gets +1.5 reward — a strong signal that real hardware normal is safe


# ── MODEL DEFINITIONS (exact copies — do NOT change) ─────────────────────────
#these must be identical to rl_train.py and the other scripts
#otherwise saved weights won't load correctly

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.scale     = math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores       = (Q @ K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        out          = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out).mean(dim=1)


class CNN_Only(nn.Module):
    def __init__(self, feat_dim=12, seq_len=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * seq_len, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.fc(self.conv(x.transpose(1, 2))).squeeze(1)


class CNN_Attention(nn.Module):
    def __init__(self, feat_dim=12, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.fc(self.multi_head_attn(h)).squeeze(1)


class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self, feat_dim=12, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        h, _ = self.bilstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        return self.fc(self.multi_head_attn(h)).squeeze(1)


# ── Q-NETWORK (identical to rl_train.py) ─────────────────────────────────────

class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256),       torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )
        #the Q-network takes a 16-dim state and outputs 16 Q-values (one per action)
        #the Q-value for an action represents its expected total future reward
        #the agent always picks the action with the highest Q-value
    def forward(self, x):
        return self.net(x)


# ── REPLAY BUFFER (identical to rl_train.py) ──────────────────────────────────

class Replay:
    def __init__(self, cap):
        self.cap = cap
        #maximum number of experiences stored
        self.s  = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.a  = np.zeros((cap,),           dtype=np.int64)
        self.r  = np.zeros((cap,),           dtype=np.float32)
        self.ns = np.zeros((cap,STATE_DIM),  dtype=np.float32)
        self.d  = np.zeros((cap,),           dtype=np.float32)
        #s = state, a = action, r = reward, ns = next state, d = done flag
        self.i  = 0
        self.n  = 0

    def add(self, s, a, r, ns, d):
        self.s[self.i]  = s
        self.a[self.i]  = a
        self.r[self.i]  = r
        self.ns[self.i] = ns
        self.d[self.i]  = float(d)
        self.i = (self.i + 1) % self.cap
        #circular buffer: when we reach the end we wrap back to position 0
        #and start overwriting the oldest experiences
        self.n = min(self.n + 1, self.cap)
        #track how many experiences are actually stored (up to the capacity)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        return (
            torch.tensor(self.s[idx]),
            torch.tensor(self.a[idx]),
            torch.tensor(self.r[idx]),
            torch.tensor(self.ns[idx]),
            torch.tensor(self.d[idx]),
        )
        #randomly sample a mini-batch from the replay buffer
        #random sampling breaks the correlation between consecutive experiences
        #which makes training more stable


# ── HARDWARE NORMAL BUFFER ────────────────────────────────────────────────────

class HardwareNormalBuffer:
    """
    Reads real hardware pcap, builds state vectors for every normal sequence,
    and stores (state, action=ALLOW, reward=HW_ALLOW_REWARD, next_state, done=0)
    tuples so the DQN replay can sample from them.

    We build a fake 'next state' by shifting the window by 1 sequence — good
    enough for the RL to learn the ALLOW policy on this kind of traffic.
    """
    #this buffer holds experiences generated entirely from real hardware normal traffic
    #each experience says: "in this real hardware state, the correct action is ALLOW"
    #by mixing these into every training batch we teach the agent to not block real sensors

    def __init__(self, pcap_path: Path, scaler, detector, device: str):
        self.device    = device
        self.tuples    = []   # list of (s, a, r, ns, done) numpy arrays
        self._build(pcap_path, scaler, detector)

    # ── feature extraction (mirrors live code exactly) ────────────────────────
    @staticmethod
    def _pkt_to_feat(pkt, prev_time):
        feat = np.zeros(FEAT_DIM, dtype=np.float32)
        try:
            tcp   = pkt[TCP]
            sport = int(tcp.sport)
            dport = int(tcp.dport)
            if sport not in MQTT_PORTS and dport not in MQTT_PORTS:
                return None, prev_time
            #skip packets that don't involve MQTT ports — same filter as training

            t      = float(pkt.time)
            length = float(pkt.len) if hasattr(pkt, "len") else float(len(pkt))
            t_delta = (t - prev_time) if prev_time is not None else 0.0

            to_mqtt   = int(dport in MQTT_PORTS)
            from_mqtt = int(sport in MQTT_PORTS)
            flags_int = int(tcp.flags)
            #extract all TCP flag bits using bitwise AND
            #for example 0x02 is the SYN bit so & 0x02 != 0 means SYN is set

            feat[0]  = t
            feat[1]  = t_delta
            feat[2]  = length
            feat[3]  = int(to_mqtt or from_mqtt)
            feat[4]  = int(flags_int & 0x02 != 0)  # SYN
            feat[5]  = int(flags_int & 0x10 != 0)  # ACK
            feat[6]  = int(flags_int & 0x01 != 0)  # FIN
            feat[7]  = int(flags_int & 0x04 != 0)  # RST
            feat[8]  = int(flags_int & 0x08 != 0)  # PSH
            feat[9]  = int(flags_int & 0x20 != 0)  # URG
            feat[10] = to_mqtt
            feat[11] = from_mqtt
        except Exception:
            return None, prev_time
        return feat, t

    def _scale_seq(self, seq, scaler):
        s = seq.copy()
        s[:, 0] = s[:, 0] - s[:, 0].min()   # relative time inside window
        s[:, [0, 1, 2]] = scaler.transform(s[:, [0, 1, 2]])
        return s
        #scale the sequence using the fine-tuned scaler from 4a

    def _seq_to_state(self, seq_scaled, det_p):
        """Build the same 16-dim state vector as LiveStateTracker.build_state."""
        time_delta_mean = float(np.mean(seq_scaled[:, 1]))
        length_mean     = float(np.mean(seq_scaled[:, 2]))
        has_mqtt_mean   = float(np.mean(seq_scaled[:, 3]))
        flags_mean      = np.mean(seq_scaled[:, 4:12], axis=0).astype(np.float32)
        extras = np.array([
            det_p,
            time_delta_mean,
            length_mean,
            has_mqtt_mean,
            0.0,   # prev_action = ALLOW (0/15) — hardware is always in a clean state
            0.0,   # escalation_level = 0 — no escalation for normal hardware traffic
            0.0,   # fp_counter = 0 — no recent false alarms assumed
            0.0,   # fn_counter = 0 — no recent misses assumed
        ], dtype=np.float32)
        return np.concatenate([extras, flags_mean]).astype(np.float32)
        #build the same 16-dim state vector as the environment so the agent
        #can't tell the difference between a hardware experience and a dataset experience

    def _build(self, pcap_path: Path, scaler, detector):
        print(f"[HW-BUFFER] Reading {pcap_path} ...")

        try:
            df = pd.read_csv(pcap_path)
        except Exception as e:
            print(f"[HW-BUFFER] Could not read hardware CSV: {e}")
            return
        #read the real hardware CSV safely — if it fails we just skip hardware experiences

        df.columns = [c.strip() for c in df.columns]
        required_cols = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
        if any(c not in df.columns for c in required_cols):
            print(f"[HW-BUFFER] Missing required columns in hardware CSV.")
            print(f"[HW-BUFFER] Available columns: {list(df.columns)}")
            return

        df = df.loc[is_tcp_mqtt_port_df(df), required_cols].copy()
        if df.empty:
            print("[HW-BUFFER] No TCP rows with MQTT port found in hardware CSV.")
            return
        #apply the same filter as everything else — only keep TCP+MQTT port rows

        df["label"] = 0
        df["file_name"] = pcap_path.name
        df = df.sort_values("Time").reset_index(drop=True)
        #all hardware rows get label=0 because this is captured normal traffic
        #sort by time so the sequence order is correct

        feats = _add_features_one_file(df)
        #extract the 12 features from the hardware dataframe

        print(f"[HW-BUFFER] {len(feats):,} MQTT-TCP packets extracted")
        if len(feats) < SEQ_LEN + STEP:
            print("[HW-BUFFER] Not enough packets — hardware buffer will be empty.")
            return

        detector.eval()
        seqs = []
        for i in range(0, len(feats) - SEQ_LEN + 1, STEP):
            seqs.append(np.stack(feats[i : i + SEQ_LEN], axis=0))
        #build overlapping windows of 20 packets from the hardware data

        for idx in range(len(seqs) - 1):   # -1 so we always have a next_seq
            seq    = seqs[idx]
            next_s = seqs[idx + 1]
            #we need a current sequence and the next one so we can build (state, next_state) pairs
            #this is the standard experience replay format: (s, a, r, s')

            seq_sc  = self._scale_seq(seq,    scaler)
            next_sc = self._scale_seq(next_s, scaler)

            # detector probability for this sequence
            with torch.no_grad():
                x     = torch.tensor(seq_sc[None, ...], dtype=torch.float32)
                det_p = float(torch.sigmoid(detector(x)).item())
                #run the fine-tuned detector to get the attack probability for this hardware sequence

                x_next     = torch.tensor(next_sc[None, ...], dtype=torch.float32)
                det_p_next = float(torch.sigmoid(detector(x_next)).item())
                #also get the detector probability for the next sequence
                #this goes into the next_state so the Q-network can estimate future value

            state      = self._seq_to_state(seq_sc,  det_p)
            next_state = self._seq_to_state(next_sc, det_p_next)

            self.tuples.append((
                state.astype(np.float32),
                A_ALLOW,
                #the correct action for all hardware normal experiences is ALLOW
                HW_ALLOW_REWARD,
                #we give a strong positive reward of +1.5 for ALLOWing real normal traffic
                next_state.astype(np.float32),
                0.0,   # not done — the episode continues after this step
            ))

        print(f"[HW-BUFFER] Built {len(self.tuples):,} hardware-normal experience tuples")

    def sample(self, n: int):
        """Sample n random tuples from the hardware buffer."""
        if len(self.tuples) == 0:
            return None
        idx  = np.random.randint(0, len(self.tuples), size=n)
        s    = np.stack([self.tuples[i][0] for i in idx])
        a    = np.array([self.tuples[i][1] for i in idx], dtype=np.int64)
        r    = np.array([self.tuples[i][2] for i in idx], dtype=np.float32)
        ns   = np.stack([self.tuples[i][3] for i in idx])
        d    = np.array([self.tuples[i][4] for i in idx], dtype=np.float32)
        return (
            torch.tensor(s),
            torch.tensor(a),
            torch.tensor(r),
            torch.tensor(ns),
            torch.tensor(d),
        )
        #randomly sample n tuples and return as tensors ready for the Q-network update

    def __len__(self):
        return len(self.tuples)


# ── HELPER FUNCTIONS (identical to rl_train.py) ───────────────────────────────

def eps_by_step(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_START + (EPS_END - EPS_START) * (step / EPS_DECAY_STEPS)
    #compute the current epsilon for epsilon-greedy exploration
    #at step 0 epsilon=1.0 so 100% random actions
    #by step 50,000 epsilon=0.05 so 95% greedy and 5% random
    #linear decay between those two points


def evaluate_greedy_true_metrics(env, qnet, episodes=5, seed=None):
    qnet.eval()
    total_reward = 0.0
    TP = TN = FP = FN = 0
    action_counts = Counter()

    if seed is not None:
        rng_state = np.random.get_state()
        py_state  = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        #fix the random seed so validation always uses the same episodes
        #this makes val metrics comparable across training runs

    try:
        for _ in range(episodes):
            s    = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    qs = qnet(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                    a  = int(torch.argmax(qs, dim=1).item())
                #greedy action — always pick the highest Q-value action (no exploration)
                ns, r, done, info = env.step(a)
                s = ns
                total_reward += r
                action_counts[info["action_name"]] += 1

                y           = int(info["y_true"])
                act         = int(info["action"])
                is_allow    = (act == A_ALLOW)
                is_mitigate = (act not in NON_MITIGATE_ACTIONS)

                if y == 1:
                    TP += 1 if is_mitigate else 0
                    FN += 1 if not is_mitigate else 0
                    #TP: attack correctly mitigated
                    #FN: attack allowed through or only soft-handled
                else:
                    TN += 1 if is_allow else 0
                    FP += 1 if not is_allow else 0
                    #TN: normal traffic correctly allowed
                    #FP: normal traffic wrongly blocked or restricted
    finally:
        if seed is not None:
            np.random.set_state(rng_state)
            random.setstate(py_state)
            #restore the original random state so training continues normally

    attack_total = TP + FN
    normal_total = TN + FP
    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    qnet.train()
    return {
        "val_reward": float(total_reward / max(episodes, 1)),
        "TP": int(TP), "FN": int(FN), "TN": int(TN), "FP": int(FP),
        "accuracy":  (TP + TN) / max(TP + TN + FP + FN, 1),
        "precision": precision, "recall": recall, "f1": f1,
        "TPR": TP / max(attack_total, 1),
        "TNR": TN / max(normal_total, 1),
        "FPR": FP / max(normal_total, 1),
        "FNR": FN / max(attack_total, 1),
        "top_actions": action_counts,
    }


def is_acceptable(vm):
    return (vm["FPR"] <= MAX_FPR and vm["FNR"] <= MAX_FNR
            and vm["TPR"] >= MIN_TPR and vm["TNR"] >= MIN_TNR)
    #a model is accepted only if it meets all four safety thresholds at once
    #FPR and FNR define how safe it is to users and to security
    #TPR and TNR confirm it performs well on both attack and normal traffic


def save_top3_checkpoints(q, score, ep, top3, paths):
    top3.append((float(score), int(ep), None))
    top3.sort(key=lambda x: x[0], reverse=True)
    top3 = top3[:3]
    #keep only the top 3 best-scoring models
    for i, (sc, e, _) in enumerate(top3):
        torch.save(q.state_dict(), paths[i])
        top3[i] = (sc, e, str(paths[i]))
    return top3
    #we always save the 3 best checkpoints from training
    #this is useful if the very best model was reached at a middle episode


def append_csv_row(csv_path: Path, header, row_dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row_dict)
    #append one row to the training log CSV
    #we write one row per episode so we can analyse training progress later


def build_detector(model_type):
    if model_type == "cnn_only":
        return CNN_Only(feat_dim=FEAT_DIM, seq_len=SEQ_LEN)
    elif model_type == "cnn_attention":
        return CNN_Attention(feat_dim=FEAT_DIM, num_heads=4)
    elif model_type == "cnn_bilstm_attn":
        return CNN_BiLSTM_Attn(feat_dim=FEAT_DIM, num_heads=4)
    raise ValueError(f"Unknown model type: {model_type}")


# ── MAIN TRAINING FUNCTION ────────────────────────────────────────────────────

def train_single_detector(detector_name, detector_config, data):
    print(f"\n{'='*80}")
    print(f"[INFO] Mixed RL re-training for detector: {detector_name}")
    print(f"{'='*80}\n")

    DETECTOR_CKPT = detector_config["ckpt"]
    SCALER_CKPT   = detector_config["scaler"]
    DETECTOR_TYPE = detector_config["type"]

    # output files — suffix _ft so originals are untouched
    RL_BEST1         = OUT_DIR / f"rl_dqn_preventer_best_{detector_name}_ft.pt"
    RL_BEST2         = OUT_DIR / f"rl_dqn_preventer_rank2_{detector_name}_ft.pt"
    RL_BEST3         = OUT_DIR / f"rl_dqn_preventer_rank3_{detector_name}_ft.pt"
    RL_ACCEPTED_BEST = OUT_DIR / f"rl_policy_best_ACCEPTED_{detector_name}_ft.pt"
    RL_OVERALL_BEST  = OUT_DIR / f"rl_policy_best_OVERALL_{detector_name}_ft.pt"
    RL_LOG_CSV       = OUT_DIR / f"rl_training_metrics_{detector_name}_ft.csv"
    #all output files use the _ft suffix so the original RL models are never overwritten
    #ACCEPTED is the model that passed all safety thresholds — this is what we deploy
    #OVERALL is the model with the best overall score even if it failed some thresholds

    Xtr, ytr, ftr = data["train"]["X"], data["train"]["y"], data["train"]["files"]
    Xva, yva, fva = data["val"]["X"],   data["val"]["y"],   data["val"]["files"]

    env = NeuroGuardRLEnv(
        X_seq=Xtr, y_seq=ytr, files_seq=ftr,
        detector_ckpt=DETECTOR_CKPT, scaler_ckpt=SCALER_CKPT,
        detector_type=DETECTOR_TYPE, device=DEVICE,
        max_steps_per_episode=MAX_STEPS_EP,
    )
    #training environment uses the train split and the fine-tuned detector from 4a
    val_env = NeuroGuardRLEnv(
        X_seq=Xva, y_seq=yva, files_seq=fva,
        detector_ckpt=DETECTOR_CKPT, scaler_ckpt=SCALER_CKPT,
        detector_type=DETECTOR_TYPE, device=DEVICE,
        max_steps_per_episode=MAX_STEPS_EP,
    )
    #separate validation environment on the val split — the agent never trains on these episodes

    # load fine-tuned detector for the hardware buffer
    print(f"[INFO] Loading fine-tuned detector for hardware buffer: {DETECTOR_CKPT}")
    hw_detector = build_detector(DETECTOR_TYPE)
    hw_detector.load_state_dict(torch.load(DETECTOR_CKPT, map_location="cpu"))
    hw_detector.eval()
    #we need a copy of the detector to compute det_p for the hardware buffer sequences
    #it runs on CPU because the buffer is built once before training starts

    with open(SCALER_CKPT, "rb") as f:
        hw_scaler = pickle.load(f)
    #load the fine-tuned scaler so hardware sequences are scaled consistently

    # build the hardware normal experience buffer
    hw_buf = HardwareNormalBuffer(PCAP_FILE, hw_scaler, hw_detector, DEVICE)
    has_hw = len(hw_buf) > 0
    if has_hw:
        print(f"[INFO] Hardware buffer ready: {len(hw_buf):,} normal tuples (30% of each batch)")
    else:
        print("[WARNING] Hardware buffer is empty — training on dataset only (check pcap path)")
    #if the hardware CSV is missing or empty we fall back to training on dataset only
    #this is a graceful degradation so training can still run

    # DQN networks
    q  = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
    #main Q-network — this is the one we train and eventually deploy
    tq = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
    tq.load_state_dict(q.state_dict())
    #target Q-network — a delayed copy of q used to compute stable training targets
    #without a target network the Q-values would chase themselves and training would diverge
    opt     = torch.optim.Adam(q.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    #SmoothL1Loss is also called Huber loss — it behaves like L2 for small errors
    #but like L1 for large errors so it is less sensitive to outliers than MSE
    replay  = Replay(REPLAY_SIZE)
    #the dataset experience replay buffer

    global_step        = 0
    top3               = []
    best_overall_score = -1e9
    best_accepted_score= -1e9
    best_accepted_ep   = None

    header = [
        "episode", "train_reward", "val_reward", "eps", "global_step",
        "TPR", "TNR", "FPR", "FNR", "TP", "TN", "FP", "FN",
        "val_score", "accepted", "accepted_score",
        "top1_action", "top1_count", "top2_action", "top2_count", "top3_action", "top3_count",
    ]
    if RL_LOG_CSV.exists():
        RL_LOG_CSV.unlink()
    #delete the old log file so we start fresh for this training run

    print(f"[INFO] Starting DQN training — {EPISODES} episodes...")

    for ep in range(1, EPISODES + 1):
        s         = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            eps = eps_by_step(global_step)

            if random.random() < eps:
                a = random.randint(0, N_ACTIONS - 1)
                #exploration: take a random action so we discover new behaviours
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                    a  = int(torch.argmax(qs, dim=1).item())
                #exploitation: take the best known action according to current Q-values

            ns, r, done, _ = env.step(a)
            replay.add(s, a, r, ns, done)
            s           = ns
            ep_reward  += r
            global_step += 1

            if replay.n >= START_LEARNING:
                # ── mixed batch: 70% dataset + 30% hardware ────────────────
                hw_n   = int(BATCH * HW_BATCH_FRAC) if has_hw else 0
                ds_n   = BATCH - hw_n
                #hw_n is the number of hardware experiences per batch (30% = ~77 of 256)
                #ds_n is the number of dataset experiences per batch (70% = ~179 of 256)

                bs, ba, br, bns, bd = replay.sample(ds_n)
                #sample the dataset portion from the standard replay buffer

                if hw_n > 0:
                    hw_sample = hw_buf.sample(hw_n)
                    if hw_sample is not None:
                        hs, ha, hr, hns, hd = hw_sample
                        bs  = torch.cat([bs,  hs],  dim=0)
                        ba  = torch.cat([ba,  ha],  dim=0)
                        br  = torch.cat([br,  hr],  dim=0)
                        bns = torch.cat([bns, hns], dim=0)
                        bd  = torch.cat([bd,  hd],  dim=0)
                        #concatenate hardware experiences onto the dataset batch
                        #so the combined batch has both types of experiences

                bs  = bs.to(DEVICE)
                ba  = ba.to(DEVICE)
                br  = br.to(DEVICE)
                bns = bns.to(DEVICE)
                bd  = bd.to(DEVICE)

                qsa = q(bs).gather(1, ba.view(-1, 1)).squeeze(1)
                #gather the Q-value for the action that was actually taken
                with torch.no_grad():
                    best_a = torch.argmax(q(bns), dim=1)
                    next_q = tq(bns).gather(1, best_a.view(-1, 1)).squeeze(1)
                    target = br + GAMMA * (1.0 - bd) * next_q
                #Double DQN: the main network picks the best next action
                #the target network evaluates its Q-value
                #this reduces overestimation compared to vanilla DQN

                loss = loss_fn(qsa, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if global_step % TARGET_UPDATE == 0:
                    tq.load_state_dict(q.state_dict())
                    #every 1,000 steps sync the target network with the main network

        # ── validation ────────────────────────────────────────────────────
        vm         = evaluate_greedy_true_metrics(val_env, q, episodes=VAL_EVAL_EPISODES, seed=VAL_SEED)
        eps_now    = eps_by_step(global_step)
        val_score  = (0.55 * vm["TPR"]) + (0.55 * vm["TNR"]) - (0.70 * vm["FPR"]) - (0.30 * vm["FNR"])
        #custom scoring formula that prioritises reducing false positives (FPR weight 0.70)
        #we care most about not blocking real sensors so FPR gets the heaviest penalty
        accepted   = is_acceptable(vm)
        acc_score  = (vm["TPR"] + vm["TNR"]) - (vm["FPR"] + vm["FNR"])
        #secondary score used to rank accepted models among themselves

        tops = vm["top_actions"].most_common(3)
        while len(tops) < 3:
            tops.append(("NONE", 0))

        status = "ACCEPT" if accepted else "REJECT"
        print(
            f"[{detector_name}] EP {ep:03d} | train_reward={ep_reward:.2f} | val_reward={vm['val_reward']:.2f} "
            f"| eps={eps_now:.3f} | TPR={vm['TPR']:.3f} TNR={vm['TNR']:.3f} "
            f"FPR={vm['FPR']:.3f} FNR={vm['FNR']:.3f} | score={val_score:.4f} | {status} | top={tops[0][0]}"
        )

        append_csv_row(RL_LOG_CSV, header, {
            "episode": ep, "train_reward": float(ep_reward), "val_reward": float(vm["val_reward"]),
            "eps": float(eps_now), "global_step": int(global_step),
            "TPR": float(vm["TPR"]), "TNR": float(vm["TNR"]),
            "FPR": float(vm["FPR"]), "FNR": float(vm["FNR"]),
            "TP": int(vm["TP"]), "TN": int(vm["TN"]), "FP": int(vm["FP"]), "FN": int(vm["FN"]),
            "val_score": float(val_score), "accepted": int(accepted), "accepted_score": float(acc_score),
            "top1_action": str(tops[0][0]), "top1_count": int(tops[0][1]),
            "top2_action": str(tops[1][0]), "top2_count": int(tops[1][1]),
            "top3_action": str(tops[2][0]), "top3_count": int(tops[2][1]),
        })
        #log all metrics for this episode to the CSV so we can plot training curves later

        if val_score > best_overall_score:
            best_overall_score = val_score
            torch.save(q.state_dict(), RL_OVERALL_BEST)
            print(f"  [OK] saved best overall: {RL_OVERALL_BEST} | score={best_overall_score:.4f}")

        if accepted and acc_score > best_accepted_score:
            best_accepted_score = acc_score
            best_accepted_ep    = ep
            torch.save(q.state_dict(), RL_ACCEPTED_BEST)
            print(
                f"  [ACCEPTED] saved: {RL_ACCEPTED_BEST} | ep={ep} "
                f"| TPR={vm['TPR']:.3f} TNR={vm['TNR']:.3f} FPR={vm['FPR']:.3f} FNR={vm['FNR']:.3f}"
            )
            #this is the model we will deploy — it met all safety thresholds AND
            #it is the best accepted model seen so far in training
        elif not accepted:
            print(f"  not accepted (need: TPR≥{MIN_TPR}, TNR≥{MIN_TNR}, FPR≤{MAX_FPR}, FNR≤{MAX_FNR})")

        paths = [RL_BEST1, RL_BEST2, RL_BEST3]
        if (len(top3) < 3) or (val_score > top3[-1][0]):
            top3 = save_top3_checkpoints(q, val_score, ep, top3, paths)
            print(f"  [OK] updated Top-3 for {detector_name}:")
            for i, (sc, e, p) in enumerate(top3, 1):
                print(f"    rank{i}: score={sc:.4f} ep={e} file={p}")

    print(f"\n[DONE] Mixed RL re-training finished for {detector_name}.")
    if best_accepted_ep:
        print(f"Best ACCEPTED model at episode {best_accepted_ep} → {RL_ACCEPTED_BEST}")
    else:
        print("No model met the safety thresholds. Use the best-overall model.")
    print(f"Log: {RL_LOG_CSV}")

    return {
        "detector_name":      detector_name,
        "best_overall_score": best_overall_score,
        "best_accepted_score":best_accepted_score if best_accepted_ep else None,
        "best_accepted_ep":   best_accepted_ep,
    }


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixed RL re-training with hardware normal data")
    parser.add_argument(
        "--detector", type=str,
        choices=["cnn_only", "cnn_attn", "cnn_bilstm_attn", "all"],
        default="all",
        help="Which detector to train with (default: all)",
    )
    args = parser.parse_args()
    #command-line argument lets us train one specific detector or all three at once
    #useful for testing one model quickly without waiting for all three to finish

    if not PCAP_FILE.exists():
        print(f"[ERROR] CSV not found: {PCAP_FILE}")
        print("        Update PCAP_FILE at the top of this script.")
        exit(1)

    print("[INFO] Building RL sequences from dataset...")
    data = build_rl_data_from_csv(
        CSV_ALL, TRAIN_IDX, VAL_IDX, TEST_IDX,
        seq_len=SEQ_LEN, step=STEP
    )
    #load all sequences from the original simulation dataset (train+val splits)
    #these come from the big labeled CSV from step 1 using the index files from step 2

    all_results = []
    detectors_to_train = (
        list(DETECTOR_CONFIGS.items())
        if args.detector == "all"
        else [(args.detector, DETECTOR_CONFIGS[args.detector])]
    )

    for det_name, det_cfg in detectors_to_train:
        result = train_single_detector(det_name, det_cfg, data)
        all_results.append(result)
        #train the RL policy for each selected detector with the mixed experience approach

    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON")
        print("=" * 80)
        for r in all_results:
            print(f"\n{r['detector_name']}:")
            print(f"  Best Overall Score  : {r['best_overall_score']:.4f}")
            if r["best_accepted_score"] is not None:
                print(f"  Best Accepted Score : {r['best_accepted_score']:.4f}  (Episode {r['best_accepted_ep']})")
            else:
                print(f"  Best Accepted Score : None (no model met thresholds)")
        best = max(all_results, key=lambda x: x["best_overall_score"])
        print(f"\nBEST DETECTOR: {best['detector_name']}  score={best['best_overall_score']:.4f}")

    print("\n[INFO] Next step: copy the _ft model files to your Raspberry Pi model directory.")
    #after this step the _ft RL policy files are ready to be copied to the Pi for deployment