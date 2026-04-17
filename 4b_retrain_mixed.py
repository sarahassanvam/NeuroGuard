#4b retrain the rl policy using the fine-tuned detector and both hardware normal and hardware attack buffers
import random
import csv
import json
import math
import pickle
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List
#importing tools for csv logging, json loading, file paths, argument parsing, and type hints

import numpy as np
import torch
from torch import nn
import pandas as pd
#importing numpy for arrays, pytorch for the neural networks, and pandas for reading csvs

OUT_DIR        = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")
#the main folder that holds the dataset, split indices, saved detector weights and rl policy outputs
HW_NORMAL_CSV  = Path(r"C:\Users\User\Downloads\broker_data_transfer.csv")
#path to the hardware normal traffic csv (real mqtt sessions from the raspberry pi)
HW_ATTACK_CSV  = Path(r"C:\Users\User\Downloads\hping3_traffic.csv")
#path to the hardware attack traffic csv (hping3 syn flood captured on real hardware)

CSV_ALL   = OUT_DIR / "mqtt_packets_labeled.csv"
#the full original labeled dataset used to build the rl training and validation sequences
TRAIN_IDX = OUT_DIR / "train_idx.npy"
VAL_IDX   = OUT_DIR / "val_idx.npy"
TEST_IDX  = OUT_DIR / "test_idx.npy"
#saved row indices for train, val and test splits to avoid data leakage

DETECTOR_CONFIGS = {
    "cnn_only": {
        "ckpt":   OUT_DIR / "detector_cnn_only_ft2.pt",
        #path to the fine-tuned cnn-only detector weights from 4a
        "scaler": OUT_DIR / "scaler_cnn_only_ft2.pkl",
        #path to the refitted scaler from 4a
        "thresh": OUT_DIR / "threshold_cnn_only_ft2.json",
        #path to the calibrated decision threshold from 4a
        "type":   "cnn_only",
        #the model architecture type so we build the right class
    },
    "cnn_attn": {
        "ckpt":   OUT_DIR / "detector_cnn_attention_ft2.pt",
        "scaler": OUT_DIR / "scaler_cnn_attention_ft2.pkl",
        "thresh": OUT_DIR / "threshold_cnn_attention_ft2.json",
        "type":   "cnn_attention",
    },
    "cnn_bilstm_attn": {
        "ckpt":   OUT_DIR / "detector_cnn_bilstm_attn_ft2.pt",
        "scaler": OUT_DIR / "scaler_cnn_bilstm_attn_ft2.pkl",
        "thresh": OUT_DIR / "threshold_cnn_bilstm_attn_ft2.json",
        "type":   "cnn_bilstm_attn",
    },
}
#this dictionary maps each detector name to its fine-tuned checkpoint, scaler, threshold and architecture type

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
#use gpu if available otherwise fall back to cpu
SEQ_LEN         = 20
#each sample is a sliding window of 20 consecutive packets exactly like training
STEP            = 5
#we slide the window forward by 5 packets each time to get overlapping sequences
FEAT_DIM        = 12
#each packet has 12 features: Time, time_delta, Length, has_mqtt_port, 6 tcp flags, to_mqtt, from_mqtt
MQTT_PORTS      = {1883, 8883}
#the two standard mqtt ports we filter on
N_ACTIONS       = 16
#the rl agent can choose from 16 different actions (from ALLOW to ISOLATE_NODE)
STATE_DIM       = 16
#the state vector that the rl agent receives has 16 values describing the current situation
EPISODES        = 225
#train the rl agent for 225 episodes (each episode = one file of sequences)
MAX_STEPS_EP    = 200
#each episode can have at most 200 steps (sequences) to keep training time manageable
GAMMA           = 0.99
#discount factor: 0.99 means future rewards matter almost as much as immediate ones
LR              = 1e-3
#learning rate for the q-network optimizer
BATCH           = 256
#how many experiences to sample from the replay buffer in each learning step
REPLAY_SIZE     = 200_000
#the replay buffer stores up to 200,000 past experiences before overwriting old ones
START_LEARNING  = 5_000
#the agent doesn't start learning until 5,000 experiences are collected so the buffer isn't too empty
TARGET_UPDATE   = 1_000
#copy the main q-network weights to the target network every 1,000 steps to stabilize learning
EPS_START       = 1.0
#start with fully random actions so the agent explores the environment first
EPS_END         = 0.05
#end with 5% random actions and 95% policy actions so the agent mostly exploits what it learned
EPS_DECAY_STEPS = 50_000
#linearly reduce randomness from EPS_START to EPS_END over 50,000 steps
VAL_EVAL_EPISODES = 30
#evaluate on 30 validation episodes after each training episode for a stable signal
VAL_SEED        = 123
#fixed seed so validation episodes are the same every time for fair comparison

MAX_FPR = 0.10
#the model must have a false positive rate at or below 10% to be accepted
MAX_FNR = 0.10
#the model must have a false negative rate at or below 10% to be accepted
MIN_TPR = 0.90
#the model must detect at least 90% of real attacks to be accepted
MIN_TNR = 0.90
#the model must correctly allow at least 90% of normal traffic to be accepted

HW_NORMAL_BATCH_FRAC  = 0.15
#15% of each training batch comes from the hardware normal buffer so the agent sees ALLOW rewarded often
HW_ATTACK_BATCH_FRAC  = 0.05
#5% of each training batch comes from the hardware attack buffer so the agent learns to mitigate hardware attacks

HW_NORMAL_SKIP_ABOVE  = 0.50
#skip hardware normal windows where the detector thinks it's an attack to avoid confusing the agent
HW_ATTACK_SKIP_BELOW  = 0.20
#skip hardware attack windows where the detector is already very confident it's normal to avoid teaching the agent to block normal traffic

HW_NORMAL_REWARD      = 2.0
#reward given to the agent when it correctly allows hardware normal traffic
HW_ATTACK_REWARD      = 2.5
#reward given when the agent correctly mitigates hardware attack traffic (slightly higher to prioritize detection)
DET_NORMAL_LOW_THR    = 0.40
#below this detector probability we are confident the traffic is normal

HW_ATTACK_DEFAULT_ACTION = 4
#default mitigation action for hping3 syn floods which is DROP_SYN since that directly counters tcp syn packets

A_ALLOW          = 0
#action 0: let traffic pass without any intervention
A_RATE_LIMIT     = 1
#action 1: limit the request rate from this source
A_TEMP_BLOCK     = 2
#action 2: temporarily block the ip or client
A_PERM_BLOCK     = 3
#action 3: permanently block the ip or client
A_DROP_SYN       = 4
#action 4: drop tcp syn packets to stop syn flooding attacks
A_DROP_CONNECT   = 5
#action 5: drop connection attempts
A_DELAY_CONNECT  = 6
#action 6: delay connection attempts (slows down flooding)
A_LIMIT_PUBLISH  = 7
#action 7: limit mqtt publish messages from this client
A_BLOCK_SUBSCRIBE= 8
#action 8: block mqtt subscribe messages from this client
A_DISCONNECT     = 9
#action 9: force the client to disconnect
A_QUARANTINE     = 10
#action 10: isolate the client from normal network traffic
A_ISOLATE_NODE   = 11
#action 11: fully isolate this node at the network level
A_REDUCE_QOS     = 12
#action 12: reduce the mqtt quality of service level for this client
A_ALERT_ONLY     = 13
#action 13: raise an alert without taking any blocking action
A_ESCALATE       = 14
#action 14: increase the response severity level
A_DEESCALATE     = 15
#action 15: decrease the response severity level

ACTION_NAMES = {
    0: "ALLOW",           1: "RATE_LIMIT_IP",      2: "TEMP_BLOCK_IP",
    3: "PERM_BLOCK_IP",   4: "DROP_SYN_DELAY_TCP",  5: "DROP_CONNECT",
    6: "DELAY_CONNECT",   7: "LIMIT_PUBLISH",       8: "BLOCK_SUBSCRIBE",
    9: "DISCONNECT_CLIENT", 10: "QUARANTINE_CLIENT", 11: "ISOLATE_NODE",
    12: "REDUCE_QOS",     13: "ALERT_ONLY",          14: "ESCALATE",
    15: "DEESCALATE",
}
#maps each action id to a human-readable name for logging and debugging

NON_MITIGATE_ACTIONS = {A_ALLOW, A_ALERT_ONLY, A_ESCALATE, A_DEESCALATE}
#actions that do not directly stop an attack so if the agent takes one during an attack it counts as a false negative
SAFE_ACTIONS         = {A_ALLOW, A_ALERT_ONLY, A_DEESCALATE}
#actions that are safe on normal traffic so they don't count as false positives
HEAVY_ACTIONS        = {A_TEMP_BLOCK, A_PERM_BLOCK, A_DISCONNECT, A_QUARANTINE, A_ISOLATE_NODE}
#heavy actions that can break normal users so they should only be used when the agent is very confident

FEATURE_COLS = [
    "Time", "time_delta", "Length",
    "has_mqtt_port",
    "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
    "to_mqtt", "from_mqtt",
]
#the ordered list of all 12 features used to build the raw feature matrix from each packet

from Preventer.rl_env import NeuroGuardRLEnv, build_rl_data_from_csv
#import the rl environment and data builder from the original rl_env module


class MultiHeadAttention(nn.Module):
    #multi-head attention that lets the model focus on the most informative packets from multiple angles
    def __init__(self, hidden_dim, num_heads=4):
        #hidden_dim is the feature size per packet after cnn processing
        #num_heads is how many independent attention patterns to learn
        super().__init__()
        assert hidden_dim % num_heads == 0
        #hidden_dim must divide evenly across all heads or the split is impossible
        self.num_heads = num_heads
        #store the number of heads
        self.head_dim  = hidden_dim // num_heads
        #each head processes this many features
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        #query projection: what this packet is looking for
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        #key projection: what information this packet contains
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        #value projection: what information this packet provides when attended to
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        #combines all heads back into one vector
        self.scale     = math.sqrt(self.head_dim)
        #scaling factor to prevent dot products from getting too large

    def forward(self, x):
        #x shape is (batch, seq_len, hidden_dim)
        B, T, D = x.shape
        #unpack the three dimensions: batch, timesteps, features
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        #compute queries and reshape so each head handles its own slice of features
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        #compute keys with the same reshape
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        #compute values with the same reshape
        scores       = (Q @ K.transpose(-2, -1)) / self.scale
        #attention scores: how much each packet should attend to every other packet
        attn_weights = torch.softmax(scores, dim=-1)
        #convert scores to probabilities that sum to 1 so attention is a weighted average
        out          = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, D)
        #apply attention weights to values and reassemble all heads into one tensor
        return self.out_proj(out).mean(dim=1)
        #project the combined output then average across all timesteps to get one vector per sequence


class CNN_Only(nn.Module):
    #simple cnn-only detector that finds short-term local patterns in packet sequences
    def __init__(self, feat_dim=12, seq_len=20):
        #feat_dim is the number of features per packet and seq_len is the number of packets per window
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer looks at 3 packets at a time and learns 128 local patterns
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
            #second conv layer goes deeper on those 128 channels
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            #flatten 128 channels x seq_len timesteps into one long vector
            nn.Linear(128 * seq_len, 128), nn.ReLU(), nn.Dropout(0.3),
            #compress with dropout to prevent overfitting
            nn.Linear(128, 1),
            #output one logit for attack vs normal
        )
    def forward(self, x):
        return self.fc(self.conv(x.transpose(1, 2))).squeeze(1)
        #transpose input for conv1d then run through cnn and fc layers


class CNN_Attention(nn.Module):
    #cnn combined with multi-head attention to focus on the most important packets
    def __init__(self, feat_dim=12, num_heads=4):
        #feat_dim is the number of features per packet and num_heads is the number of attention heads
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer finds local attack patterns in groups of 3 packets
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
            #second conv layer builds richer representations from the first layer's output
        )
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        #attention sits on top of cnn output to focus on the most informative steps
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        #compress and make the final attack vs normal decision
    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        #run cnn then transpose back to (batch, timesteps, channels) for attention
        return self.fc(self.multi_head_attn(h)).squeeze(1)
        #pass through attention then fc layers and squeeze the output


class CNN_BiLSTM_Attn(nn.Module):
    #the most powerful model: cnn for local patterns + bilstm for sequential context + attention for focusing
    def __init__(self, feat_dim=12, num_heads=4):
        #feat_dim is the number of features per packet and num_heads is the number of attention heads
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer detects local burst and flag patterns
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
            #second conv layer learns deeper local representations
        )
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        #bilstm reads the sequence both forward and backward to capture dependencies in both directions
        #hidden size is 64, bidirectional doubles output to 128
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        #attention focuses on the most informative steps in the bilstm output
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        #compress the attention output and produce the final attack vs normal decision
    def forward(self, x):
        h, _ = self.bilstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        #run cnn then bilstm: h contains forward and backward hidden states at every step
        return self.fc(self.multi_head_attn(h)).squeeze(1)
        #pass bilstm output through attention then fc and squeeze the extra dimension


class QNet(nn.Module):
    #the q-network is the rl agent's brain that maps a state to q-values for all 16 actions
    def __init__(self, state_dim, n_actions):
        #state_dim is the length of the state vector (16) and n_actions is the number of possible actions (16)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            #expand from 16 state values to 256 neurons to have enough capacity to learn complex action policies
            nn.Linear(256, 256),       nn.ReLU(),
            #a second 256-neuron layer for deeper pattern recognition
            nn.Linear(256, n_actions),
            #output one q-value per action so the agent picks the action with the highest q-value
        )
    def forward(self, x):
        return self.net(x)
        #pass the state vector through all three layers and return one q-value per action


class Replay:
    #the replay buffer stores past experiences so the agent can learn from them later in random batches
    def __init__(self, cap):
        #cap is the maximum number of experiences the buffer can hold before overwriting old ones
        self.cap = cap
        #store the capacity
        self.s   = np.zeros((cap, STATE_DIM), dtype=np.float32)
        #pre-allocate array for states: shape is (capacity, state_dim)
        self.a   = np.zeros((cap,),           dtype=np.int64)
        #pre-allocate array for actions taken
        self.r   = np.zeros((cap,),           dtype=np.float32)
        #pre-allocate array for rewards received
        self.ns  = np.zeros((cap, STATE_DIM), dtype=np.float32)
        #pre-allocate array for next states (state after the action)
        self.d   = np.zeros((cap,),           dtype=np.float32)
        #pre-allocate array for done flags (1.0 if the episode ended else 0.0)
        self.i   = 0
        #write pointer: points to where the next experience will be written
        self.n   = 0
        #how many experiences are currently stored (increases until it hits cap)

    def add(self, s, a, r, ns, d):
        #this function adds one experience (s, a, r, ns, d) to the buffer
        self.s[self.i]  = s
        #write the current state
        self.a[self.i]  = a
        #write the action that was taken
        self.r[self.i]  = r
        #write the reward that was received
        self.ns[self.i] = ns
        #write the next state after the action
        self.d[self.i]  = float(d)
        #write the done flag converted to float (1.0=done, 0.0=not done)
        self.i = (self.i + 1) % self.cap
        #advance the write pointer and wrap around when it reaches the end
        self.n = min(self.n + 1, self.cap)
        #increase the stored count up to the maximum capacity

    def sample(self, batch):
        #this function randomly picks a batch of experiences from the buffer for learning
        idx = np.random.randint(0, self.n, size=batch)
        #randomly pick batch many indices from the valid stored range
        return (
            torch.tensor(self.s[idx]),
            #return states as a pytorch tensor
            torch.tensor(self.a[idx]),
            #return actions as a pytorch tensor
            torch.tensor(self.r[idx]),
            #return rewards as a pytorch tensor
            torch.tensor(self.ns[idx]),
            #return next states as a pytorch tensor
            torch.tensor(self.d[idx]),
            #return done flags as a pytorch tensor
        )


def _is_tcp_mqtt_port(df: pd.DataFrame) -> pd.Series:
    #this function checks each row to see if it is a tcp packet on an mqtt port
    proto = df.get("Protocol", pd.Series([""] * len(df))).astype(str).str.upper()
    #get the Protocol column as uppercase for consistent matching
    info  = df.get("Info",     pd.Series([""] * len(df))).astype(str).str.upper()
    #get the Info column as uppercase so port number matching is case-insensitive
    return (proto == "TCP") & info.str.contains(r"\b1883\b|\b8883\b", regex=True, na=False)
    #return True only if the packet is TCP and its info field contains mqtt port 1883 or 8883


def _add_features(df: pd.DataFrame) -> np.ndarray:
    #this function builds the 12-feature matrix from a hardware csv dataframe
    g = df.copy()
    #work on a copy so we never modify the original dataframe
    g["Time"]   = pd.to_numeric(g["Time"],   errors="coerce").fillna(0.0)
    #convert Time to numeric and replace parsing errors with 0
    g["Length"] = pd.to_numeric(g["Length"], errors="coerce").fillna(0.0)
    #convert Length to numeric and replace parsing errors with 0

    g["Time"]       = g["Time"] - g["Time"].min()
    #normalize Time at the file level by subtracting the minimum timestamp so it starts from 0
    g["time_delta"] = g["Time"].diff().fillna(0.0)
    #compute the time difference between consecutive packets and fill the first one with 0

    info  = g["Info"].astype(str).str.upper()
    #convert the Info column to uppercase for consistent port and flag matching
    ports = info.str.extract(r"(\d+)\s*[>→]\s*(\d+)")
    #extract source and destination ports from info strings like "51781 → 1883"
    sport = pd.to_numeric(ports[0], errors="coerce")
    #convert source port to numeric
    dport = pd.to_numeric(ports[1], errors="coerce")
    #convert destination port to numeric

    g["has_mqtt_port"] = ((sport.isin(MQTT_PORTS)) | (dport.isin(MQTT_PORTS))).fillna(False).astype(int)
    #1 if either source or destination is an mqtt port otherwise 0
    g["to_mqtt"]       = (dport.isin(MQTT_PORTS)).fillna(False).astype(int)
    #1 if the packet is going toward the mqtt broker
    g["from_mqtt"]     = (sport.isin(MQTT_PORTS)).fillna(False).astype(int)
    #1 if the packet is coming from the mqtt broker
    g["flag_syn"]      = info.str.contains(r"\bSYN\b", regex=True, na=False).astype(int)
    #1 if the syn flag is present in the info field
    g["flag_ack"]      = info.str.contains(r"\bACK\b", regex=True, na=False).astype(int)
    #1 if the ack flag is present
    g["flag_fin"]      = info.str.contains(r"\bFIN\b", regex=True, na=False).astype(int)
    #1 if the fin flag is present
    g["flag_rst"]      = info.str.contains(r"\bRST\b", regex=True, na=False).astype(int)
    #1 if the rst flag is present
    g["flag_psh"]      = info.str.contains(r"\bPSH\b", regex=True, na=False).astype(int)
    #1 if the psh flag is present
    g["flag_urg"]      = info.str.contains(r"\bURG\b", regex=True, na=False).astype(int)
    #1 if the urg flag is present

    return g[FEATURE_COLS].to_numpy(dtype=np.float32)
    #return the 12-feature matrix as a float32 numpy array in the correct column order


def _scale_seq(seq: np.ndarray, scaler) -> np.ndarray:
    #this function applies the scaler only to the three continuous features (Time, time_delta, Length)
    s = seq.copy()
    #work on a copy so we never modify the original sequence
    s[:, [0, 1, 2]] = scaler.transform(s[:, [0, 1, 2]])
    #scale only columns 0, 1, 2 (binary features in columns 3-11 are never scaled)
    return s
    #return the scaled sequence


def _seq_to_state(seq_scaled: np.ndarray, det_p: float,
                  prev_action: int, escalation_level: int,
                  fp_counter: int, fn_counter: int) -> np.ndarray:
    #this function builds the 16-dimensional rl state vector from a scaled sequence and agent context
    time_delta_mean    = float(np.mean(seq_scaled[:, 1]))
    #mean inter-packet time across the 20-packet window (scaled)
    length_mean        = float(np.mean(seq_scaled[:, 2]))
    #mean packet size across the window (scaled)
    has_mqtt_port_mean = float(np.mean(seq_scaled[:, 3]))
    #fraction of packets in the window that involve an mqtt port
    flags_mean         = np.mean(seq_scaled[:, 4:12], axis=0).astype(np.float32)
    #mean value of each of the 8 flag and direction features across all 20 packets in the window

    extras = np.array([
        det_p,
        #detector attack probability for this sequence (0=confident normal, 1=confident attack)
        time_delta_mean,
        #mean inter-packet time already defined above
        length_mean,
        #mean packet length already defined above
        has_mqtt_port_mean,
        #mqtt port presence fraction already defined above
        float(prev_action)       / 15.0,
        #last action taken normalized to 0-1 so the agent knows what it just did
        float(escalation_level)  / 5.0,
        #current escalation level normalized to 0-1 (5 is maximum escalation) means how strong the current response has become
        float(fp_counter)        / 10.0,
        #false positive pressure normalized to 0-1 (grows when the agent blocks normal traffic)
        float(fn_counter)        / 10.0,
        #false negative pressure normalized to 0-1 (grows when the agent misses attacks)
    ], dtype=np.float32)

    return np.concatenate([extras, flags_mean]).astype(np.float32)
    #concatenate the 8 scalar features with the 8 flag means to produce the 16-dimensional state vector


class HardwareNormalBuffer:
    #pre-builds (state, ALLOW, reward, next_state, done=0) tuples from hardware normal traffic
    #this teaches the rl agent to reward itself for allowing clean hardware traffic

    def __init__(self, hw_csv_path: Path, scaler, detector, detector_type: str,
                 calibrated_thr: float = 0.50):
        #hw_csv_path is the path to the hardware normal csv
        #scaler is the refitted scaler from 4a
        #detector is the fine-tuned detector model
        #calibrated_thr is the threshold from 4a used to skip ambiguous windows
        self.tuples: List = []
        #empty list that will hold all valid (s, a, r, ns, d) tuples
        self._build(hw_csv_path, scaler, detector, detector_type, calibrated_thr)
        #immediately build all tuples when the buffer is created

    def _build(self, hw_csv_path: Path, scaler, detector, detector_type: str,
               calibrated_thr: float):
        #this function reads the hardware csv and converts it into a list of rl experience tuples
        print(f"[HW-NORMAL-BUFFER] Reading {hw_csv_path} ...")
        df = pd.read_csv(hw_csv_path)
        #read the hardware normal csv into memory
        df.columns = [c.strip() for c in df.columns]
        #strip whitespace from column names

        required = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
        #all six of these columns must be present for feature extraction to work
        if any(c not in df.columns for c in required):
            raise RuntimeError(f"[HW-NORMAL-BUFFER] Missing columns in {hw_csv_path}")
        #safety check: raise an error immediately if any required column is missing

        df   = df.loc[_is_tcp_mqtt_port(df), required].copy()
        #filter to only tcp mqtt rows to match the training pipeline
        df   = df.sort_values("Time").reset_index(drop=True)
        #sort by time to ensure order
        feats = _add_features(df)
        #build the 12-feature matrix for all packets
        print(f"[HW-NORMAL-BUFFER] {len(feats):,} packets after filtering")

        if len(feats) < SEQ_LEN + STEP:
            print("[HW-NORMAL-BUFFER] WARNING: Not enough packets. Buffer will be empty.")
            return
        #safety check: if fewer packets than one window we cannot build any tuples

        seqs = [feats[i:i + SEQ_LEN] for i in range(0, len(feats) - SEQ_LEN + 1, STEP)]
        #build all sliding window sequences from the hardware normal packets

        detector.eval()
        #switch the detector to evaluation mode so dropout is disabled
        skipped = 0
        #count how many windows we skip because the detector thinks they are attacks
        kept    = 0
        #count how many windows we actually keep as valid tuples

        prev_action      = A_ALLOW
        #start the agent context assuming it was allowing traffic (makes sense for normal data)
        escalation_level = 0
        #start with no escalation since this is normal traffic
        fp_counter       = 0
        #start with no false positive pressure
        fn_counter       = 0
        #start with no false negative pressure

        for idx in range(len(seqs) - 1):
            #go through all windows except the last one cuz for the last window, there is no window after it, so there is no next state.
            seq      = seqs[idx]
            #the current window of 20 packets
            next_seq = seqs[idx + 1]
            #the following window which becomes the next state after the agent takes an action

            seq_sc      = _scale_seq(seq,      scaler)
            #scale the current window's continuous features
            next_seq_sc = _scale_seq(next_seq, scaler)
            #scale the next window's continuous features

            with torch.no_grad():
                x      = torch.tensor(seq_sc[None, ...],      dtype=torch.float32)
                det_p  = float(torch.sigmoid(detector(x)).item())
                #run the detector on the current window to get an attack probability
                xn     = torch.tensor(next_seq_sc[None, ...], dtype=torch.float32)
                det_pn = float(torch.sigmoid(detector(xn)).item())
                #run the detector on the next window to build the next state

            if det_p >= calibrated_thr:
                skipped += 1
                continue
            #skip windows where the detector already thinks it is an attack since those are ambiguous for a normal buffer

            hw_reward = HW_NORMAL_REWARD if det_p < DET_NORMAL_LOW_THR else HW_NORMAL_REWARD * 0.5
            #give full reward if the detector is very confident it is normal otherwise give half reward

            state      = _seq_to_state(seq_sc,      det_p,  prev_action, escalation_level, fp_counter, fn_counter)
            #build the current state vector from the scaled window and current agent context
            next_state = _seq_to_state(next_seq_sc, det_pn, A_ALLOW,     escalation_level,
                                       max(fp_counter - 1, 0), fn_counter)
            #build the next state assuming the agent took ALLOW and fp_counter decreased by 1

            self.tuples.append((
                state.astype(np.float32),
                #the current state
                A_ALLOW,
                #the correct action for normal traffic is always ALLOW
                float(hw_reward),
                #the reward for allowing this normal window
                next_state.astype(np.float32),
                #the next state after allowing
                0.0,
                #done=0.0 because we are not at the end of an episode
            ))

            prev_action = A_ALLOW
            #update the context: the last action taken was ALLOW
            fp_counter  = max(fp_counter - 1, 0)
            #decrease fp pressure since we correctly allowed normal traffic (floor at 0)
            kept += 1
            #count this window as kept

        print(f"[HW-NORMAL-BUFFER] Windows: {len(seqs):,} | "
              f"Skipped (det_p>={calibrated_thr:.2f}): {skipped:,} | Kept: {kept:,}")
        #show how many windows were processed, skipped and kept

    def sample(self, n: int):
        #this function returns n randomly selected tuples as pytorch tensors for batch training
        if len(self.tuples) == 0:
            return None
        #if the buffer is empty return None so the caller knows nothing is available
        idx = np.random.randint(0, len(self.tuples), size=n)
        #randomly pick n indices from the stored tuples
        s   = np.stack([self.tuples[i][0] for i in idx])
        #stack the selected states into a batch array
        a   = np.array([self.tuples[i][1] for i in idx], dtype=np.int64)
        #stack the selected actions into an array
        r   = np.array([self.tuples[i][2] for i in idx], dtype=np.float32)
        #stack the selected rewards
        ns  = np.stack([self.tuples[i][3] for i in idx])
        #stack the selected next states
        d   = np.array([self.tuples[i][4] for i in idx], dtype=np.float32)
        #stack the selected done flags
        return (torch.tensor(s), torch.tensor(a), torch.tensor(r),
                torch.tensor(ns), torch.tensor(d))
        #return all five as pytorch tensors ready for the learning step

    def __len__(self):
        return len(self.tuples)
        #return how many tuples are stored so the caller can check if the buffer is populated


class HardwareAttackBuffer:
    #pre-builds (state, MITIGATE, reward, next_state, done=0) tuples from hardware attack traffic
    #this teaches the rl agent to mitigate real hping3 syn flood attacks from hardware

    def __init__(self, hw_csv_path: Path, scaler, detector, detector_type: str,
                 calibrated_thr: float = 0.50):
        #hw_csv_path is the path to the hardware attack csv
        #scaler is the refitted scaler from 4a
        #detector is the fine-tuned detector model
        #calibrated_thr is the threshold from 4a used to skip ambiguous windows
        self.tuples: List = []
        #empty list to hold all valid attack mitigation tuples
        self._build(hw_csv_path, scaler, detector, detector_type, calibrated_thr)
        #immediately build all tuples when the buffer is created

    def _build(self, hw_csv_path: Path, scaler, detector, detector_type: str,
               calibrated_thr: float):
        #this function reads the hardware attack csv and builds rl experience tuples for mitigation
        print(f"[HW-ATTACK-BUFFER] Reading {hw_csv_path} ...")
        df = pd.read_csv(hw_csv_path)
        #read the hardware attack csv
        df.columns = [c.strip() for c in df.columns]
        #strip whitespace from column names

        required = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
        #all six columns must be present for feature extraction
        if any(c not in df.columns for c in required):
            raise RuntimeError(f"[HW-ATTACK-BUFFER] Missing columns in {hw_csv_path}")
        #safety check: raise an error if any required column is missing

        df_raw = df.copy()
        #save the raw dataframe before filtering so the fallback below has access to all rows

        df = df.loc[_is_tcp_mqtt_port(df), required].copy()
        #filter to tcp mqtt rows matching the training pipeline
        df = df.sort_values("Time").reset_index(drop=True)
        #sort by time for chronological order

        if df.empty:
            #hping3 may produce a different arrow format in the Info column depending on the tshark version
            print("[HW-ATTACK-BUFFER] MQTT-port filter returned empty.")
            info_col   = df_raw.get("Info",     pd.Series([""] * len(df_raw))).astype(str)
            port_mask  = info_col.str.contains(r"1883|8883", regex=True, na=False)
            #try a broader search without word boundaries in case the arrow format is different
            proto_mask = df_raw.get("Protocol", pd.Series([""] * len(df_raw))).astype(str).str.upper() == "TCP"
            df = df_raw.loc[proto_mask & port_mask, required].copy()
            df = df.sort_values("Time").reset_index(drop=True)
            #try the broader filter: tcp + port number appears anywhere in info

            if df.empty:
                print("[HW-ATTACK-BUFFER] WARNING: No MQTT-port rows found even with "
                      "broad search. Falling back to all TCP rows. Verify your hping3 "
                      "CSV contains packets to port 1883 and the Info column includes "
                      "'1883' or '8883'. Check with: grep -c '1883' your_file.csv")
                df = df_raw.loc[proto_mask, required].copy()
                df = df.sort_values("Time").reset_index(drop=True)
                #last resort fallback: accept all tcp rows if no mqtt port rows are found at all

        feats = _add_features(df)
        #build the 12-feature matrix from the filtered attack packets
        print(f"[HW-ATTACK-BUFFER] {len(feats):,} packets after filtering")

        if len(feats) < SEQ_LEN + STEP:
            print("[HW-ATTACK-BUFFER] WARNING: Not enough packets. Buffer will be empty.")
            return
        #safety check: if fewer packets than one window we cannot build any tuples

        seqs = [feats[i:i + SEQ_LEN] for i in range(0, len(feats) - SEQ_LEN + 1, STEP)]
        #build all sliding window sequences from the attack packets

        detector.eval()
        #switch the detector to evaluation mode
        skipped = 0
        #count how many windows we skip because the detector is already very confident they are normal
        kept    = 0
        #count how many windows we keep as valid attack mitigation tuples

        prev_action      = A_DROP_SYN
        #start the context from a mitigation action since we are dealing with attack data
        escalation_level = 1
        #mild escalation since an attack is in progress
        fp_counter       = 0
        #no false positive pressure since we know this is attack traffic
        fn_counter       = 2
        #some false negative pressure to encourage the agent to mitigate

        for idx in range(len(seqs) - 1):
            #go through all windows except the last one
            seq      = seqs[idx]
            #current window of 20 attack packets
            next_seq = seqs[idx + 1]
            #next window which becomes the next state

            seq_sc      = _scale_seq(seq,      scaler)
            #scale the current window
            next_seq_sc = _scale_seq(next_seq, scaler)
            #scale the next window

            with torch.no_grad():
                x      = torch.tensor(seq_sc[None, ...],      dtype=torch.float32)
                det_p  = float(torch.sigmoid(detector(x)).item())
                #get detector probability for the current window
                xn     = torch.tensor(next_seq_sc[None, ...], dtype=torch.float32)
                det_pn = float(torch.sigmoid(detector(xn)).item())
                #get detector probability for the next window

            if det_p < HW_ATTACK_SKIP_BELOW:
                skipped += 1
                continue
            #skip windows where the detector is already very confident it is normal to avoid teaching the agent to block innocent traffic

            hw_reward = HW_ATTACK_REWARD if det_p >= calibrated_thr else HW_ATTACK_REWARD * 0.6
            #give full reward if the detector confirms it is an attack otherwise give partial reward for uncertain windows

            mitigate_action = HW_ATTACK_DEFAULT_ACTION
            #use DROP_SYN as the default action since hping3 generates syn floods

            state      = _seq_to_state(seq_sc,      det_p,  prev_action, escalation_level, fp_counter, fn_counter)
            #build the current state vector
            next_state = _seq_to_state(next_seq_sc, det_pn, mitigate_action, escalation_level,
                                       fp_counter, max(fn_counter - 1, 0))
            #build the next state assuming the mitigation action was taken and fn_counter decreased by 1

            self.tuples.append((
                state.astype(np.float32),
                #the current state
                mitigate_action,
                #the correct action for this attack window (DROP_SYN)
                float(hw_reward),
                #the reward for mitigating this attack
                next_state.astype(np.float32),
                #the next state after mitigation
                0.0,
                #done=0.0 since we are not at the end of an episode
            ))

            prev_action = mitigate_action
            #update context: the last action taken was DROP_SYN
            fn_counter  = max(fn_counter - 1, 0)
            #decrease fn pressure since we mitigated an attack (floor at 0)
            kept += 1
            #count this window as kept

        print(f"[HW-ATTACK-BUFFER] Windows: {len(seqs):,} | "
              f"Skipped (det_p<{HW_ATTACK_SKIP_BELOW:.2f}): {skipped:,} | Kept: {kept:,}")
        #show how many windows were processed, skipped and kept

    def sample(self, n: int):
        #this function returns n randomly selected attack mitigation tuples as pytorch tensors
        if len(self.tuples) == 0:
            return None
        #if the buffer is empty return None
        idx = np.random.randint(0, len(self.tuples), size=n)
        #randomly pick n indices
        s   = np.stack([self.tuples[i][0] for i in idx])
        #stack states into a batch
        a   = np.array([self.tuples[i][1] for i in idx], dtype=np.int64)
        #stack actions
        r   = np.array([self.tuples[i][2] for i in idx], dtype=np.float32)
        #stack rewards
        ns  = np.stack([self.tuples[i][3] for i in idx])
        #stack next states
        d   = np.array([self.tuples[i][4] for i in idx], dtype=np.float32)
        #stack done flags
        return (torch.tensor(s), torch.tensor(a), torch.tensor(r),
                torch.tensor(ns), torch.tensor(d))
        #return all five as pytorch tensors

    def __len__(self):
        return len(self.tuples)
        #return how many tuples are stored


def eps_by_step(step):
    #this function computes the current epsilon (exploration rate) based on how many steps have been taken
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    #once we have taken EPS_DECAY_STEPS steps we stop decaying and stay at EPS_END
    return EPS_START + (EPS_END - EPS_START) * (step / EPS_DECAY_STEPS)
    #linearly interpolate from EPS_START down to EPS_END over the first EPS_DECAY_STEPS steps


def evaluate_greedy_true_metrics(env, qnet, episodes=5, seed=None):
    #this function evaluates the agent greedily (no exploration) and computes true tp, tn, fp, fn counts
    qnet.eval()
    #switch to evaluation mode so dropout is disabled
    total_reward = 0.0
    #accumulate total reward across all episodes
    TP = TN = FP = FN = 0
    #initialize confusion counts
    action_counts = Counter()
    #track which actions the agent takes during evaluation

    if seed is not None:
        rng_state = np.random.get_state()
        #save the current numpy random state so we can restore it after evaluation
        py_state  = random.getstate()
        #save the current python random state
        np.random.seed(seed)
        #set numpy seed for reproducible episode ordering
        random.seed(seed)
        #set python seed for reproducible episode ordering

    try:
        for _ in range(episodes):
            #run one episode per iteration
            s    = env.reset()
            #reset the environment to the start of a new file
            done = False
            while not done:
                with torch.no_grad():
                    qs = qnet(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                    a  = int(torch.argmax(qs, dim=1).item())
                #compute q-values for all actions and pick the best one (greedy, no randomness)
                ns, r, done, info = env.step(a)
                #apply the chosen action and get the next state, reward and done flag
                s = ns
                #update the current state
                total_reward += r
                #add this step's reward to the running total
                action_counts[info["action_name"]] += 1
                #count how often this action name was chosen

                y           = int(info["y_true"])
                #get the true label for this sequence (0=normal, 1=attack)
                act         = int(info["action"])
                #get the action id that was taken
                is_allow    = (act == A_ALLOW)
                #true if the agent chose to allow the traffic
                is_mitigate = (act not in NON_MITIGATE_ACTIONS)
                #true if the agent chose a real mitigation action (not just alert or escalate)

                if y == 1:
                    TP += 1 if is_mitigate else 0
                    #attack that was mitigated correctly counts as true positive
                    FN += 1 if not is_mitigate else 0
                    #attack that was not mitigated counts as false negative
                else:
                    TN += 1 if is_allow else 0
                    #normal traffic that was allowed counts as true negative
                    FP += 1 if not is_allow else 0
                    #normal traffic that was blocked counts as false positive
    finally:
        if seed is not None:
            np.random.set_state(rng_state)
            #restore numpy random state so training randomness is not affected
            random.setstate(py_state)
            #restore python random state

    attack_total = TP + FN
    #total number of attack sequences seen across all evaluation episodes
    normal_total = TN + FP
    #total number of normal sequences seen
    precision    = TP / max(TP + FP, 1)
    #of all sequences the agent mitigated how many were actually attacks
    recall       = TP / max(TP + FN, 1)
    #of all attack sequences how many did the agent mitigate
    f1           = 2 * precision * recall / max(precision + recall, 1e-9)
    #harmonic mean of precision and recall

    qnet.train()
    #switch back to training mode after evaluation
    return {
        "val_reward": float(total_reward / max(episodes, 1)),
        #average reward per episode
        "TP": int(TP), "FN": int(FN), "TN": int(TN), "FP": int(FP),
        #raw confusion counts
        "accuracy":   (TP + TN) / max(TP + TN + FP + FN, 1),
        #fraction of all decisions that were correct
        "precision":  precision,
        #see above
        "recall":     recall,
        #see above
        "f1":         f1,
        #see above
        "TPR": TP / max(attack_total, 1),
        #true positive rate: fraction of attacks correctly mitigated
        "TNR": TN / max(normal_total, 1),
        #true negative rate: fraction of normal traffic correctly allowed
        "FPR": FP / max(normal_total, 1),
        #false positive rate: fraction of normal traffic wrongly blocked
        "FNR": FN / max(attack_total, 1),
        #false negative rate: fraction of attacks that slipped through
        "top_actions": action_counts,
        #counter of which actions were most used during evaluation
    }


def is_acceptable(vm):
    return (vm["FPR"] <= MAX_FPR and vm["FNR"] <= MAX_FNR
            and vm["TPR"] >= MIN_TPR and vm["TNR"] >= MIN_TNR)
    #return True only if all four safety thresholds are satisfied so we only save truly safe models


def append_csv_row(csv_path: Path, header, row_dict):
    #this function appends one row of training metrics to the log csv file
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    #create the directory if it doesn't exist yet
    exists = csv_path.exists()
    #check if the csv already exists so we know whether to write the header
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        #open the csv in append mode so we add to it without overwriting previous rows
        w = csv.DictWriter(f, fieldnames=header)
        #create a dictionary writer that uses the header list as column names
        if not exists:
            w.writeheader()
        #if the file is new write the header row first
        w.writerow(row_dict)
        #write this episode's metrics as one row


def build_detector(model_type):
    #this function builds the correct detector architecture based on the type string
    if model_type == "cnn_only":
        return CNN_Only(feat_dim=FEAT_DIM, seq_len=SEQ_LEN)
    #build the simple cnn-only detector
    elif model_type == "cnn_attention":
        return CNN_Attention(feat_dim=FEAT_DIM, num_heads=4)
    #build the cnn + attention detector
    elif model_type == "cnn_bilstm_attn":
        return CNN_BiLSTM_Attn(feat_dim=FEAT_DIM, num_heads=4)
    #build the full cnn + bilstm + attention detector
    raise ValueError(f"Unknown model type: {model_type}")
    #safety check: raise an error for unknown types


def train_single_detector(detector_name, detector_config, data):
    #this function runs the full dqn training loop for one detector and returns a summary
    print(f"\n{'='*80}")
    print(f"[INFO] Mixed RL re-training (FT2) for detector: {detector_name}")
    print(f"{'='*80}\n")
    #print a separator so the output for each detector is easy to identify

    DETECTOR_CKPT = detector_config["ckpt"]
    #path to the fine-tuned detector weights
    SCALER_CKPT   = detector_config["scaler"]
    #path to the refitted scaler
    THRESH_FILE   = detector_config["thresh"]
    #path to the calibrated threshold json
    DETECTOR_TYPE = detector_config["type"]
    #the model architecture type string

    RL_BEST_ACCEPTED = OUT_DIR / f"rl_policy_best_ACCEPTED_{detector_name}_ft2.pt"
    #where we save the rl policy that met all safety thresholds with the highest accepted score
    RL_BEST_OVERALL  = OUT_DIR / f"rl_policy_best_OVERALL_{detector_name}_ft2.pt"
    #where we save the rl policy with the highest overall validation score regardless of thresholds
    RL_LOG_CSV       = OUT_DIR / f"rl_training_metrics_{detector_name}_ft2.csv"
    #where we save the per-episode training log for analysis

    if not DETECTOR_CKPT.exists():
        print(f"[SKIP] Detector checkpoint not found: {DETECTOR_CKPT}")
        print(f"       Run 4a_transfer_learning_ft2.py first.")
        return None
    #safety check: if the fine-tuned detector does not exist yet tell the user to run 4a first

    calibrated_thr = 0.50
    #default threshold of 0.5 in case the json file doesn't exist
    if THRESH_FILE.exists():
        with open(THRESH_FILE, "r", encoding="utf-8") as f:
            thr_data = json.load(f)
        calibrated_thr = float(thr_data.get("threshold", 0.50))
        print(f"[INFO] Loaded calibrated threshold: {calibrated_thr:.2f} from {THRESH_FILE}")
        #load the threshold saved by 4a so the rl training uses the same decision boundary as deployment
    else:
        print(f"[WARNING] Threshold file not found: {THRESH_FILE}. Using default 0.50.")
        #warn if the threshold file is missing and fall back to 0.5

    Xtr, ytr, ftr = data["train"]["X"], data["train"]["y"], data["train"]["files"]
    #unpack training sequences, labels, and file names for the rl training environment
    Xva, yva, fva = data["val"]["X"],   data["val"]["y"],   data["val"]["files"]
    #unpack validation sequences, labels, and file names for the rl validation environment

    env = NeuroGuardRLEnv(
        X_seq=Xtr, y_seq=ytr, files_seq=ftr,
        detector_ckpt=DETECTOR_CKPT, scaler_ckpt=SCALER_CKPT,
        detector_type=DETECTOR_TYPE, device=DEVICE,
        max_steps_per_episode=MAX_STEPS_EP,
        det_attack_thr=calibrated_thr,
    )
    #create the training rl environment with the fine-tuned detector and calibrated threshold
    val_env = NeuroGuardRLEnv(
        X_seq=Xva, y_seq=yva, files_seq=fva,
        detector_ckpt=DETECTOR_CKPT, scaler_ckpt=SCALER_CKPT,
        detector_type=DETECTOR_TYPE, device=DEVICE,
        max_steps_per_episode=MAX_STEPS_EP,
        det_attack_thr=calibrated_thr,
    )
    #create the validation rl environment with the same fine-tuned detector and calibrated threshold
    print(f"[INFO] RL envs created with det_attack_thr = {calibrated_thr:.2f}")

    print(f"[INFO] Loading _ft2 detector for hardware buffers: {DETECTOR_CKPT}")
    hw_detector = build_detector(DETECTOR_TYPE)
    #build the detector architecture
    hw_detector.load_state_dict(torch.load(DETECTOR_CKPT, map_location="cpu"))
    #load the fine-tuned weights
    hw_detector.eval()
    #set to evaluation mode since we only use it for inference inside the hardware buffers

    with open(SCALER_CKPT, "rb") as f:
        hw_scaler = pickle.load(f)
    #load the refitted scaler from 4a for the hardware buffer feature scaling

    print("\n[INFO] Building HW normal buffer (ALLOW tuples)...")
    hw_normal_buf = HardwareNormalBuffer(
        HW_NORMAL_CSV, hw_scaler, hw_detector, DETECTOR_TYPE, calibrated_thr
    )
    #build the hardware normal buffer: all tuples teach the agent to ALLOW hardware normal traffic

    print("\n[INFO] Building HW attack buffer (MITIGATE tuples)...")
    hw_attack_buf = HardwareAttackBuffer(
        HW_ATTACK_CSV, hw_scaler, hw_detector, DETECTOR_TYPE, calibrated_thr
    )
    #build the hardware attack buffer: all tuples teach the agent to MITIGATE hardware attack traffic

    has_hw_normal = len(hw_normal_buf) > 0
    #true if the normal buffer has at least one tuple
    has_hw_attack = len(hw_attack_buf) > 0
    #true if the attack buffer has at least one tuple

    print(f"\n[INFO] HW normal buffer tuples: {len(hw_normal_buf):,}")
    print(f"[INFO] HW attack buffer tuples: {len(hw_attack_buf):,}")
    #show buffer sizes so we can verify both buffers are populated

    q  = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
    #build the main q-network that the agent learns with
    tq = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
    #build the target q-network that provides stable learning targets
    tq.load_state_dict(q.state_dict())
    #initialize the target network with the same weights as the main network
    opt     = torch.optim.Adam(q.parameters(), lr=LR)
    #create an Adam optimizer for the main q-network
    loss_fn = nn.SmoothL1Loss()
    #use huber loss (smooth l1) which is less sensitive to outlier rewards than mse
    replay  = Replay(REPLAY_SIZE)
    #create the experience replay buffer with the configured capacity

    global_step         = 0
    #count total steps taken across all episodes
    best_overall_score  = -1e9
    #track the best overall validation score seen so far
    best_accepted_score = -1e9
    #track the best accepted validation score (only updated when the model meets all thresholds)
    best_accepted_ep    = None
    #track which episode produced the best accepted model

    header = [
        "episode", "train_reward", "val_reward", "eps", "global_step",
        "TPR", "TNR", "FPR", "FNR", "TP", "TN", "FP", "FN",
        "val_score", "accepted", "accepted_score",
        "top1_action", "top1_count", "top2_action", "top2_count", "top3_action", "top3_count",
    ]
    #define the column names for the training log csv
    if RL_LOG_CSV.exists():
        RL_LOG_CSV.unlink()
    #delete the old log file if it exists so we start fresh

    for ep in range(1, EPISODES + 1):
        #run one training episode per iteration
        s         = env.reset()
        #reset the training environment to the start of a new file
        ep_reward = 0.0
        #accumulate reward for this episode
        done      = False
        #track whether the episode is finished

        while not done:
            eps = eps_by_step(global_step)
            #compute the current exploration rate based on how many total steps have been taken

            if random.random() < eps:
                a = random.randint(0, N_ACTIONS - 1)
                #with probability eps take a random action to explore
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                    a  = int(torch.argmax(qs, dim=1).item())
                #otherwise take the greedy action (highest q-value) to exploit what we learned

            ns, r, done, _ = env.step(a)
            #apply the action and get the next state, reward and done flag
            replay.add(s, a, r, ns, done)
            #store this experience in the replay buffer
            s           = ns
            #update the current state
            ep_reward   += r
            #add this step's reward to the episode total
            global_step += 1
            #increment the global step counter

            if replay.n >= START_LEARNING:
                #only start learning once the buffer has enough experiences

                hw_n_n = int(BATCH * HW_NORMAL_BATCH_FRAC) if has_hw_normal else 0
                #how many hardware normal tuples to inject into this batch
                hw_a_n = int(BATCH * HW_ATTACK_BATCH_FRAC) if has_hw_attack else 0
                #how many hardware attack tuples to inject into this batch
                ds_n   = BATCH - hw_n_n - hw_a_n
                #the remaining batch size comes from the standard replay buffer

                bs, ba, br, bns, bd = replay.sample(ds_n)
                #sample the standard portion of the batch from the replay buffer

                if hw_n_n > 0:
                    result = hw_normal_buf.sample(hw_n_n)
                    if result is not None:
                        hs, ha, hr, hns, hd = result
                        bs  = torch.cat([bs,  hs],  dim=0)
                        ba  = torch.cat([ba,  ha],  dim=0)
                        br  = torch.cat([br,  hr],  dim=0)
                        bns = torch.cat([bns, hns], dim=0)
                        bd  = torch.cat([bd,  hd],  dim=0)
                #inject hardware normal ALLOW tuples into the batch so the agent keeps learning to allow clean hardware traffic

                if hw_a_n > 0:
                    result = hw_attack_buf.sample(hw_a_n)
                    if result is not None:
                        hs, ha, hr, hns, hd = result
                        bs  = torch.cat([bs,  hs],  dim=0)
                        ba  = torch.cat([ba,  ha],  dim=0)
                        br  = torch.cat([br,  hr],  dim=0)
                        bns = torch.cat([bns, hns], dim=0)
                        bd  = torch.cat([bd,  hd],  dim=0)
                #inject hardware attack MITIGATE tuples into the batch so the agent keeps learning to block hardware attacks

                bs  = bs.to(DEVICE)
                ba  = ba.to(DEVICE)
                br  = br.to(DEVICE)
                bns = bns.to(DEVICE)
                bd  = bd.to(DEVICE)
                #move the assembled batch to the correct device for the learning step

                qsa = q(bs).gather(1, ba.view(-1, 1)).squeeze(1)
                #get the q-value the network predicted for the action that was actually taken
                with torch.no_grad():
                    best_a = torch.argmax(q(bns), dim=1)
                    #use the main network to pick the best next action (double dqn)
                    next_q = tq(bns).gather(1, best_a.view(-1, 1)).squeeze(1)
                    #use the target network to evaluate that best next action (stable target)
                    target = br + GAMMA * (1.0 - bd) * next_q
                    #bellman equation: reward + discounted future value (zero for terminal states)

                loss = loss_fn(qsa, target)
                #compute huber loss between predicted and target q-values
                opt.zero_grad()
                #clear old gradients
                #Gradients tell the model how to change its weights to reduce the error
                loss.backward()
                #compute gradients
                #how each weight caused the error, so it knows what should change
                opt.step()
                #update the main network weights

                if global_step % TARGET_UPDATE == 0:
                    tq.load_state_dict(q.state_dict())
                #copy main network weights to the target network every TARGET_UPDATE steps

        vm      = evaluate_greedy_true_metrics(val_env, q, episodes=VAL_EVAL_EPISODES, seed=VAL_SEED)
        #evaluate the current policy greedily on validation data
        eps_now = eps_by_step(global_step)
        #get current epsilon just for logging

        hw_normal_penalty = 0.0
        #initialize the hardware normal penalty to zero before computing it
        if has_hw_normal and len(hw_normal_buf) > 0:
            n_check = min(200, len(hw_normal_buf))
            #check at most 200 hardware normal states to estimate how often the agent allows them
            sample  = hw_normal_buf.sample(n_check)
            if sample is not None:
                s_hw, _, _, _, _ = sample
                with torch.no_grad():
                    qs_hw  = q(s_hw.to(DEVICE))
                    acts_hw = torch.argmax(qs_hw, dim=1).cpu().numpy()
                hw_allow_rate = float((acts_hw == A_ALLOW).mean())
                #fraction of hardware normal states where the agent chooses ALLOW
                hw_normal_penalty = 1.0 - hw_allow_rate
                #penalty: 0 means the agent always allows hardware normal, 1 means it always blocks it
            q.train()
            #switch back to training mode after this in-place evaluation

        val_score = ((0.45 * vm["TPR"]) + (0.45 * vm["TNR"])
                     - (0.55 * vm["FPR"]) - (0.55 * vm["FNR"])
                     - (0.30 * hw_normal_penalty))
        #composite score: equal reward for tpr and tnr, equal penalty for fpr and fnr, extra penalty for blocking hardware normal traffic
        accepted  = is_acceptable(vm)
        #check if all four safety thresholds are met
        acc_score = (vm["TPR"] + vm["TNR"]) - (vm["FPR"] + vm["FNR"]) - hw_normal_penalty
        #accepted score is used to rank models that already passed the safety thresholds

        tops = vm["top_actions"].most_common(3)
        #get the 3 most frequently chosen actions during validation
        while len(tops) < 3:
            tops.append(("NONE", 0))
        #pad to 3 entries to avoid index errors when logging

        print(
            f"[{detector_name}] EP {ep:03d} | train_reward={ep_reward:.2f} | "
            f"val_reward={vm['val_reward']:.2f} | eps={eps_now:.3f} | "
            f"TPR={vm['TPR']:.3f} TNR={vm['TNR']:.3f} FPR={vm['FPR']:.3f} FNR={vm['FNR']:.3f} | "
            f"hw_allow={1-hw_normal_penalty:.3f} | "
            f"score={val_score:.4f} | {'ACCEPT' if accepted else 'REJECT'} | top={tops[0][0]}"
        )
        #print a one-line summary for this episode showing all key metrics

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
        #log all metrics for this episode to the csv file so we can analyze training progress later

        if val_score > best_overall_score:
            best_overall_score = val_score
            torch.save(q.state_dict(), RL_BEST_OVERALL)
            print(f"  [OK] saved best overall -> {RL_BEST_OVERALL.name}")
        #save the model if it achieved the best overall validation score so far

        if accepted and acc_score > best_accepted_score:
            best_accepted_score = acc_score
            best_accepted_ep    = ep
            torch.save(q.state_dict(), RL_BEST_ACCEPTED)
            print(f"  [OK] saved accepted best -> {RL_BEST_ACCEPTED.name}")
        #save the model only if it passed all safety thresholds AND had the best accepted score so far

    print(f"\n[DONE] Finished RL retraining (FT2) for {detector_name}")
    if best_accepted_ep:
        print(f"[DONE] Best ACCEPTED episode: {best_accepted_ep}")
    else:
        print("[DONE] No accepted model met thresholds. Use best-overall for inspection.")
    #report which episode produced the best accepted model or warn if no model met the thresholds

    return {
        "detector_name":       detector_name,
        #the name of the detector that was trained
        "best_overall_score":  best_overall_score,
        #the highest overall validation score achieved during training
        "best_accepted_score": best_accepted_score if best_accepted_ep else None,
        #the highest accepted score (None if no model met the thresholds)
        "best_accepted_ep":    best_accepted_ep,
        #the episode number that produced the best accepted model
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixed RL re-training — FT2 (HW Normal + HW Attack)")
    parser.add_argument(
        "--detector", type=str,
        choices=["cnn_only", "cnn_attn", "cnn_bilstm_attn", "all"],
        default="cnn_bilstm_attn",
    )
    args = parser.parse_args()
    #parse the command line argument to decide which detector to train with

    for path in [HW_NORMAL_CSV, HW_ATTACK_CSV]:
        if not path.exists():
            raise FileNotFoundError(f"Hardware CSV not found: {path}")
    #safety check: make sure both hardware csv files exist before starting training

    print("[INFO] Building RL sequences from dataset...")
    data = build_rl_data_from_csv(
        CSV_ALL, TRAIN_IDX, VAL_IDX, TEST_IDX,
        seq_len=SEQ_LEN, step=STEP,
    )
    #build train, val and test sequences from the original labeled dataset using saved split indices

    detectors_to_train = (
        list(DETECTOR_CONFIGS.items())
        if args.detector == "all"
        else [(args.detector, DETECTOR_CONFIGS[args.detector])]
    )
    #if the user asked for all detectors then train all three otherwise train only the one they specified

    all_results = []
    for det_name, det_cfg in detectors_to_train:
        result = train_single_detector(det_name, det_cfg, data)
        if result is not None:
            all_results.append(result)
    #train each selected detector and collect the results summary

    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON (FT2)")
        print("=" * 80)
        for r in all_results:
            print(f"{r['detector_name']}:")
            print(f"  Best overall score : {r['best_overall_score']:.4f}")
            if r["best_accepted_score"] is not None:
                print(f"  Best accepted score: {r['best_accepted_score']:.4f} (ep {r['best_accepted_ep']})")
            else:
                print("  Best accepted score: None")
    #if multiple detectors were trained print a side-by-side comparison of their results
