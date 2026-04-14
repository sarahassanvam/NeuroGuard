#4a transfer learning fine-tuning with both hardware normal and hardware attack data
import json
import math
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Dict
#importing tools for file reading and type hints for function inputs

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
#importing all the libraries we need for:
#loading data, building and fine-tuning the model and computing evaluation metrics

OUT_DIR        = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")
#the main folder that holds our dataset, split indices and saved model files
HW_NORMAL_CSV  = Path(r"C:\Users\User\Downloads\broker_data_transfer.csv")
#path to the hardware normal traffic csv (real raspberry pi normal mqtt traffic)
HW_ATTACK_CSV  = Path(r"C:\Users\User\Downloads\hping3_traffic.csv")
#path to the hardware attack traffic csv (real hping3 syn flood captured on hardware)

CSV_ALL   = OUT_DIR / "mqtt_packets_labeled.csv"
#the full original labeled dataset (all packets from the original training pipeline)
TRAIN_IDX = OUT_DIR / "train_idx.npy"
VAL_IDX   = OUT_DIR / "val_idx.npy"
TEST_IDX  = OUT_DIR / "test_idx.npy"
#these are the saved row numbers for train, val and test splits to avoid data leakage

SEQ_LEN    = 20
#each sample is a sliding window of 20 consecutive packets just like training
STEP       = 5
#we slide the window forward by 5 packets each time to create overlapping samples
FEAT_DIM   = 12
#each packet has 12 features: Time, time_delta, Length, has_mqtt_port, 6 tcp flags, to_mqtt, from_mqtt
MQTT_PORTS = {1883, 8883}
#the two standard mqtt ports that we filter on

FT_EPOCHS  = 10
#we fine-tune for 10 epochs which is the same as the original training
FT_LR      = 8e-5
#lower learning rate than original training so we don't overwrite what the model already learned
FT_BATCH   = 256
#how many sequences to process in one training step during fine-tuning
EVAL_BATCH = 512
#how many sequences to process in one forward pass during evaluation (no gradient needed so we can go bigger)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
#use gpu if one is available otherwise fall back to cpu

HW_NORMAL_FRAC   = 0.35
#35% of the mixed training set comes from hardware normal traffic
HW_ATTACK_FRAC   = 0.25
#25% of the mixed training set comes from hardware attack traffic (new pool added to fix oscillation)
ORIG_ATTACK_FRAC = 0.30
#30% of the mixed training set comes from original labeled attack sequences so the model remembers attack patterns
ORIG_NORMAL_FRAC = 0.10
#10% of the mixed training set comes from original normal sequences as an anchor against catastrophic forgetting

assert abs(HW_NORMAL_FRAC + HW_ATTACK_FRAC + ORIG_ATTACK_FRAC + ORIG_NORMAL_FRAC - 1.0) < 1e-9, \
    "Mixture fractions must sum to 1.0"
#safety check to make sure all four pool fractions add up to exactly 1.0 so we don't accidentally oversample

MAX_HW_NORMAL_SEQS  = 25_000
#maximum number of sequences we keep from the hardware normal pool to control memory usage
MAX_HW_ATTACK_SEQS  = 25_000
#maximum number of sequences we keep from the hardware attack pool
MAX_ATTACK_SEQS     = 30_000
#maximum number of sequences we keep from the original attack pool
MAX_NORMAL_SEQS     = 12_000
#maximum number of sequences we keep from the original normal pool

RANDOM_SEED = 42
#fixed seed so the mixing and sampling are reproducible every run

DETECTOR_CONFIGS = {
    "cnn_only": {
        "ckpt_in":      OUT_DIR / "detector_cnn_only.pt",
        #the original pre-trained cnn-only detector weights to load before fine-tuning
        "scaler_in":    OUT_DIR / "scaler_cnn_only.pkl",
        #the original scaler fitted during training that we reuse (and refit) here
        "type":         "cnn_only",
        #the model architecture type used to build the correct model class
        "ckpt_out":     OUT_DIR / "detector_cnn_only_ft2.pt",
        #where we save the fine-tuned weights after this script finishes
        "scaler_out":   OUT_DIR / "scaler_cnn_only_ft2.pkl",
        #where we save the refitted scaler (refitted on combined original + hardware data)
        "thresh_out":   OUT_DIR / "threshold_cnn_only_ft2.json",
        #where we save the calibrated decision threshold found by threshold calibration
        "results_out":  OUT_DIR / "results_transfer_cnn_only_ft2.txt",
        #where we save the final training and evaluation report for this detector
    },
    "cnn_attn": {
        "ckpt_in":      OUT_DIR / "detector_cnn_attention.pt",
        "scaler_in":    OUT_DIR / "scaler_cnn_attention.pkl",
        "type":         "cnn_attention",
        "ckpt_out":     OUT_DIR / "detector_cnn_attention_ft2.pt",
        "scaler_out":   OUT_DIR / "scaler_cnn_attention_ft2.pkl",
        "thresh_out":   OUT_DIR / "threshold_cnn_attention_ft2.json",
        "results_out":  OUT_DIR / "results_transfer_cnn_attention_ft2.txt",
    },
    "cnn_bilstm_attn": {
        "ckpt_in":      OUT_DIR / "detector_cnn_bilstm_attn.pt",
        "scaler_in":    OUT_DIR / "scaler_cnn_bilstm_attn.pkl",
        "type":         "cnn_bilstm_attn",
        "ckpt_out":     OUT_DIR / "detector_cnn_bilstm_attn_ft2.pt",
        "scaler_out":   OUT_DIR / "scaler_cnn_bilstm_attn_ft2.pkl",
        "thresh_out":   OUT_DIR / "threshold_cnn_bilstm_attn_ft2.json",
        "results_out":  OUT_DIR / "results_transfer_cnn_bilstm_attn_ft2.txt",
    },
}
#this dictionary maps each detector name to all the file paths it needs: input weights, scaler, type and output paths


class MultiHeadAttention(nn.Module):
    #multi-head attention mechanism that lets the model focus on the most important packets from multiple perspectives
    def __init__(self, hidden_dim, num_heads=4):
        #hidden_dim is the size of each packet's feature vector after cnn processing
        #num_heads is how many different attention patterns we want to learn simultaneously
        super().__init__()
        assert hidden_dim % num_heads == 0
        #safety check: hidden_dim must divide evenly into num_heads heads otherwise the split is impossible
        self.num_heads = num_heads
        #store how many heads we have
        self.head_dim  = hidden_dim // num_heads
        #each head processes this many features (hidden_dim split equally across all heads)
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        #query projection: what is this packet looking for in the sequence
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        #key projection: what information does this packet contain that others can attend to
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        #value projection: what information this packet actually provides if attended to
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        #final linear layer to combine all the heads back into one vector
        self.scale     = math.sqrt(self.head_dim)
        #scaling factor to stop dot products from getting too large which would make softmax collapse

    def forward(self, x):
        #x arrives as (batch, seq_len, hidden_dim)
        B, T, D = x.shape
        #B is batch size, T is sequence length (20 packets), D is hidden dim
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        #compute queries then reshape and transpose so each head processes its own slice
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        #compute keys with the same reshape so shape is (B, num_heads, T, head_dim)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        #compute values with the same reshape
        scores       = (Q @ K.transpose(-2, -1)) / self.scale
        #compute attention scores: how much should each packet attend to every other packet
        attn_weights = torch.softmax(scores, dim=-1)
        #convert raw scores to probabilities that sum to 1 so every packet gets a fair weight
        out          = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, D)
        #apply attention weights to values then reassemble all heads back into one tensor
        return self.out_proj(out).mean(dim=1)
        #project the combined output and average across all packets to get one vector per sequence


class CNN_Only(nn.Module):
    #simple cnn-only model that detects short-term local patterns in packet sequences
    def __init__(self, feat_dim=12, seq_len=20):
        #feat_dim is the number of features per packet (12 in our case)
        #seq_len is the number of packets in one window (20 in our case)
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer looks at 3 consecutive packets at once and learns 128 different local patterns
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
            #second conv layer goes deeper on those 128 channels and learns more complex patterns
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            #flatten all 128 channels x 20 timesteps into one long vector for the fully connected layers
            nn.Linear(128 * seq_len, 128), nn.ReLU(), nn.Dropout(0.3),
            #compress from 128*20 features down to 128 with dropout to prevent overfitting
            nn.Linear(128, 1),
            #final layer outputs one score (logit) for attack vs normal
        )

    def forward(self, x):
        #x arrives as (batch, timesteps, features) so we transpose to (batch, features, timesteps) for conv1d
        return self.fc(self.conv(x.transpose(1, 2))).squeeze(1)
        #run through cnn then fully connected layers and remove the extra dimension at the end


class CNN_Attention(nn.Module):
    #cnn for local pattern detection combined with multi-head attention to focus on important packets
    def __init__(self, feat_dim=12, num_heads=4):
        #feat_dim is the number of features per packet and num_heads is how many attention heads to use
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer scans groups of 3 packets to find local attack patterns
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
            #second conv layer goes deeper and learns more complex local representations
        )
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        #the attention layer takes the cnn output and focuses on the most important packets across the window
        self.fc = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            #compress the attention output with dropout to prevent overfitting
            nn.Linear(128, 1),
            #final decision: one logit for attack vs normal
        )

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        #run cnn on transposed input then transpose back so shape is (batch, timesteps, channels)
        return self.fc(self.multi_head_attn(h)).squeeze(1)
        #pass cnn output through attention then fully connected layers and remove the extra dimension


class CNN_BiLSTM_Attn(nn.Module):
    #the most powerful model combining cnn for local patterns + bilstm for sequential context + attention for focusing
    def __init__(self, feat_dim=12, num_heads=4):
        #feat_dim is the number of features per packet and num_heads is how many attention heads to use
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer finds short bursts and local patterns in consecutive packets
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
            #second conv layer builds on the first layer to learn richer local representations
        )
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        #bilstm reads the cnn output both forward and backward so it captures context from both past and future packets
        #input size is 128 (from cnn), hidden size is 64, bidirectional so output will be 64*2=128
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        #attention layer sits on top of bilstm and focuses on the most informative steps in the sequence
        self.fc = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            #compress the attention output and apply dropout to prevent overfitting
            nn.Linear(128, 1),
            #final output: one logit for attack vs normal
        )

    def forward(self, x):
        h, _ = self.bilstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        #run cnn then bilstm: cnn extracts local features, bilstm captures temporal dependencies across the sequence
        return self.fc(self.multi_head_attn(h)).squeeze(1)
        #pass bilstm output through attention then fully connected and remove the extra dimension


def build_model(model_type: str):
    #this function builds the right model based on the detector type string
    if model_type == "cnn_only":
        return CNN_Only(feat_dim=FEAT_DIM, seq_len=SEQ_LEN)
    #return the simple cnn-only model
    if model_type == "cnn_attention":
        return CNN_Attention(feat_dim=FEAT_DIM, num_heads=4)
    #return the cnn + multi-head attention model
    if model_type == "cnn_bilstm_attn":
        return CNN_BiLSTM_Attn(feat_dim=FEAT_DIM, num_heads=4)
    #return the full cnn + bilstm + attention model
    raise ValueError(f"Unknown model type: {model_type}")
    #safety check: if an unknown type is given raise an error immediately


class FocalLoss(nn.Module):
    #focal loss focuses training on hard ambiguous examples near the decision boundary
    #this prevents the model from collapsing to always predicting one class when the mix contains both easy and hard samples
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        #alpha=0.5 means equal weight for both classes so neither class dominates training
        #gamma=2.0 means easy examples get quadratically down-weighted so the model focuses on hard ones
        super().__init__()
        self.alpha = alpha
        #store the class balance weight
        self.gamma = gamma
        #store the focusing exponent

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        #compute standard binary cross-entropy loss per sample without reducing to a mean yet
        prob    = torch.sigmoid(logits)
        #convert raw logits to probabilities between 0 and 1
        p_t     = torch.where(targets == 1, prob, 1.0 - prob)
        #p_t is the probability of the correct class so high p_t means the model is already confident and correct
        alpha_t = torch.where(targets == 1,
                               torch.full_like(targets, self.alpha),
                               torch.full_like(targets, 1.0 - self.alpha))
        #alpha_t assigns class weights: alpha for attack samples and 1-alpha for normal samples
        focal_w = (1.0 - p_t) ** self.gamma
        #focal weight: near zero for easy confident examples and near one for hard uncertain examples
        return (alpha_t * focal_w * bce).mean()
        #multiply all three weights together and average across the batch to get one loss value


def is_tcp_mqtt_port_df(df: pd.DataFrame) -> pd.Series:
    #this function checks each row to see if it is a tcp packet on an mqtt port
    proto = df.get("Protocol", pd.Series([""] * len(df))).astype(str).str.upper()
    #get the Protocol column and convert to uppercase for consistent matching
    info  = df.get("Info",     pd.Series([""] * len(df))).astype(str).str.upper()
    #get the Info column and convert to uppercase so port numbers are easy to find
    return (proto == "TCP") & info.str.contains(r"\b1883\b|\b8883\b", regex=True, na=False)
    #return True only if the row is TCP AND the info field mentions port 1883 or 8883


def add_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    #this function builds all 12 features from the raw dataframe columns
    feat = df.copy()
    #work on a copy so we never modify the original dataframe
    feat["Time"]   = pd.to_numeric(feat["Time"],   errors="coerce").fillna(0.0)
    #convert Time column to numbers and replace any parsing errors with 0
    feat["Length"] = pd.to_numeric(feat["Length"], errors="coerce").fillna(0.0)
    #convert Length column to numbers and replace any parsing errors with 0

    feat["Time"]       = feat.groupby("file_name")["Time"].transform(lambda s: s - s.min())
    #normalize Time per file by subtracting the minimum timestamp so it starts at 0 for each capture
    feat["time_delta"] = feat.groupby("file_name")["Time"].diff().fillna(0.0)
    #compute time difference between consecutive packets within the same file and fill the first one with 0

    info  = feat["Info"].astype(str).str.upper()
    #convert the Info column to uppercase so all port and flag matching is case-insensitive
    ports = info.str.extract(r"(\d+)\s*[>→]\s*(\d+)")
    #extract source port and destination port from info strings like "51781 → 1883"
    sport = pd.to_numeric(ports[0], errors="coerce")
    #convert extracted source port to numeric and coerce any non-number to NaN
    dport = pd.to_numeric(ports[1], errors="coerce")
    #convert extracted destination port to numeric and coerce any non-number to NaN

    feat["has_mqtt_port"] = ((sport.isin(MQTT_PORTS)) | (dport.isin(MQTT_PORTS))).fillna(False).astype(int)
    #1 if either source or destination port is an mqtt port otherwise 0
    feat["to_mqtt"]       = (dport.isin(MQTT_PORTS)).fillna(False).astype(int)
    #1 if the destination port is an mqtt port meaning traffic is going toward the broker
    feat["from_mqtt"]     = (sport.isin(MQTT_PORTS)).fillna(False).astype(int)
    #1 if the source port is an mqtt port meaning traffic is coming from the broker

    feat["flag_syn"] = info.str.contains(r"\bSYN\b", regex=True, na=False).astype(int)
    #1 if the SYN flag is present in the info field (indicates connection initiation)
    feat["flag_ack"] = info.str.contains(r"\bACK\b", regex=True, na=False).astype(int)
    #1 if the ACK flag is present (acknowledgement of received data)
    feat["flag_fin"] = info.str.contains(r"\bFIN\b", regex=True, na=False).astype(int)
    #1 if the FIN flag is present (indicates connection teardown)
    feat["flag_rst"] = info.str.contains(r"\bRST\b", regex=True, na=False).astype(int)
    #1 if the RST flag is present (indicates connection reset often seen during attacks)
    feat["flag_psh"] = info.str.contains(r"\bPSH\b", regex=True, na=False).astype(int)
    #1 if the PSH flag is present (push data immediately to the application)
    feat["flag_urg"] = info.str.contains(r"\bURG\b", regex=True, na=False).astype(int)
    #1 if the URG flag is present (urgent pointer is significant)

    feature_cols = [
        "Time", "time_delta", "Length",
        "has_mqtt_port",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
        "to_mqtt", "from_mqtt",
    ]
    #this is the ordered list of all 12 features: 3 continuous + 1 port flag + 6 tcp flags + 2 direction flags
    return feat, feature_cols
    #return the dataframe with new feature columns and the ordered feature name list


def build_sequences_per_file(df_feat: pd.DataFrame, feature_cols: List[str],
                              seq_len=SEQ_LEN, step=STEP) -> Tuple[np.ndarray, np.ndarray]:
    #this function builds sliding window sequences from the feature dataframe without mixing packets from different files
    X_list, y_list = [], []
    #empty lists to collect sequences and their labels before stacking into arrays
    for fname, g in df_feat.groupby("file_name", sort=False):
        #process one capture file at a time so no packet from file A appears in a sequence from file B
        g = g.sort_values("Time")
        #sort packets by time to make sure sequences are always in the correct chronological order
        X = g[feature_cols].to_numpy(dtype=np.float32)
        #extract the feature matrix for this file as a numpy array
        y = g["label"].astype(int).to_numpy()
        #extract the labels for this file (0=normal, 1=attack)
        if len(X) < seq_len:
            continue
        #skip files that are too short to even form one complete sequence
        for start in range(0, len(X) - seq_len + 1, step):
            #slide the window from the start to the end of the file stopping when there aren't enough packets left
            end = start + seq_len
            #the window ends here so the sequence contains packets from start to end-1
            X_list.append(X[start:end])
            #save this window as one training example
            y_list.append(int(y[start:end].max()))
            #if any packet in the window is an attack then the whole window is labeled as attack
    if not X_list:
        raise RuntimeError("No sequences created. Check data and SEQ_LEN.")
    #safety check: if nothing was built it means the data was too short or empty
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)
    #combine all windows into one big array and return sequences and labels together


def scale_sequences(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    #this function scales only the continuous features (columns 0,1,2) using the fitted scaler
    Xs   = X.copy()
    #work on a copy so we never modify the original data
    flat = Xs.reshape(-1, Xs.shape[-1])
    #flatten the 3D array to 2D so we can apply the scaler which expects 2D input
    flat[:, [0, 1, 2]] = scaler.transform(flat[:, [0, 1, 2]])
    #scale only Time, time_delta, and Length columns (binary features are never scaled)
    return flat.reshape(Xs.shape).astype(np.float32)
    #reshape back to the original 3D shape and return as float32


def load_original_split_df(idx_path: Path) -> pd.DataFrame:
    #this function loads the original csv and returns only the rows belonging to a specific split
    idx = np.load(idx_path)
    #load the saved row indices for this split (train, val or test)
    df  = pd.read_csv(CSV_ALL)
    #read the full labeled csv into memory
    df.columns = [c.strip() for c in df.columns]
    #strip whitespace from column names to avoid mismatches
    needed = ["Time", "Source", "Destination", "Protocol", "Length", "Info", "label", "file_name"]
    #we only need these columns so we select them to save memory
    return df[needed].iloc[idx].copy()
    #return only the rows belonging to this split and only the columns we need


def load_hardware_csv(csv_path: Path, label: int) -> pd.DataFrame:
    #this function loads a hardware-captured csv and assigns a label (0=normal or 1=attack)
    df = pd.read_csv(csv_path)
    #read the hardware csv into memory
    df.columns = [c.strip() for c in df.columns]
    #strip whitespace from column names
    required = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
    #these six columns must exist because the feature pipeline expects them
    missing  = [c for c in required if c not in df.columns]
    #find any required columns that are missing
    if missing:
        raise RuntimeError(f"[HW CSV] Missing columns {missing} in {csv_path}")
    #if any required column is missing raise an error so we know immediately what is wrong
    df = df[required].copy()
    #keep only the required columns and work on a copy
    df = df.loc[is_tcp_mqtt_port_df(df)].copy()
    #filter to only tcp packets on mqtt ports to match the original training pipeline exactly
    if df.empty:
        raise RuntimeError(f"[HW CSV] No TCP+MQTT-port rows found in {csv_path}")
    #if filtering removed everything then the hardware csv has no usable mqtt traffic
    df["label"]     = int(label)
    #assign the given label to all rows (0 for normal hardware traffic, 1 for attack hardware traffic)
    df["file_name"] = csv_path.name
    #treat the entire capture file as one group so time normalization works correctly
    return df
    #return the filtered and labeled hardware dataframe


def subsample(X: np.ndarray, y: np.ndarray, max_keep: int) -> Tuple[np.ndarray, np.ndarray]:
    #this function randomly reduces the number of sequences to max_keep to control memory usage
    if len(X) <= max_keep:
        return X, y
    #if we already have fewer sequences than the limit then just return everything unchanged
    idx = np.random.choice(len(X), size=max_keep, replace=False)
    #randomly pick max_keep indices without replacement so no sequence is picked twice
    return X[idx], y[idx]
    #return only the selected sequences and their labels


def freeze_for_transfer(model: nn.Module, model_type: str):
    #this function freezes the cnn layers and leaves higher-level layers trainable for fine-tuning
    for p in model.conv.parameters():
        p.requires_grad = False
    #freeze all cnn parameters so the learned low-level packet features are preserved during fine-tuning

    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    #always keep the final fully connected decision layers trainable so the model can adapt its predictions

    if hasattr(model, "multi_head_attn"):
        for p in model.multi_head_attn.parameters():
            p.requires_grad = True
    #keep attention layers trainable so the model can relearn which packets matter in hardware traffic

    if hasattr(model, "bilstm"):
        for p in model.bilstm.parameters():
            p.requires_grad = True
    #keep bilstm trainable because it is most sensitive to timing differences between lab and real hardware traffic

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #count how many parameters will be updated during fine-tuning
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    #count how many parameters are locked and will not change
    print(f"  Trainable params: {trainable:,} | Frozen params: {frozen:,}")
    #print the counts so we can verify the freezing was done correctly


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict:
    #this function computes all classification metrics from true labels and predicted probabilities
    y_pred = (y_prob >= thr).astype(int)
    #convert probabilities to binary predictions using the given threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() \
        if len(np.unique(y_true)) == 2 else (0, 0, 0, 0)
    #extract tn, fp, fn, tp from the confusion matrix (handle the edge case where only one class is present)
    normal_total = tn + fp
    #total number of normal samples (used to compute fpr)
    attack_total = tp + fn
    #total number of attack samples (used to compute fnr)
    fpr = fp / max(normal_total, 1)
    #false positive rate: how often normal traffic is wrongly flagged as an attack
    fnr = fn / max(attack_total, 1)
    #false negative rate: how often real attacks are missed
    return {
        "acc":  accuracy_score(y_true, y_pred),
        #fraction of all predictions that are correct
        "prec": precision_score(y_true, y_pred, zero_division=0),
        #of all predicted attacks how many are really attacks
        "rec":  recall_score(y_true, y_pred, zero_division=0),
        #of all real attacks how many did we correctly detect
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        #harmonic mean of precision and recall so one number balances both
        "fpr":  fpr,
        #fraction of normal traffic wrongly blocked
        "fnr":  fnr,
        #fraction of real attacks that slipped through undetected
        "cm":   confusion_matrix(y_true, y_pred),
        #the full 2x2 confusion matrix for detailed inspection
    }


def predict_probs(model: nn.Module, X: np.ndarray, batch_size: int = EVAL_BATCH) -> np.ndarray:
    #this function runs the model in evaluation mode and returns predicted attack probabilities
    probs = []
    #empty list to collect batch outputs before concatenating
    model.eval()
    #switch the model to evaluation mode so dropout is disabled and predictions are deterministic
    for start in range(0, len(X), batch_size):
        #go through the data in batches to avoid running out of memory on large inputs
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=DEVICE)
        #convert this batch to a torch tensor and move it to the right device
        with torch.no_grad():
            p = torch.sigmoid(model(xb)).detach().cpu().numpy()
        #run the model without computing gradients and convert logits to probabilities
        probs.append(p)
        #save this batch's probabilities
    return np.concatenate(probs, axis=0)
    #combine all batches into one big probability array and return it


def calibrate_threshold(model: nn.Module,
                        X_hw_normal_s: np.ndarray,
                        X_hw_attack_s: np.ndarray) -> float:
    #this function sweeps thresholds on hardware data to find the best decision boundary
    X_cal = np.concatenate([X_hw_normal_s, X_hw_attack_s], axis=0)
    #combine hardware normal and hardware attack sequences into one calibration set
    y_cal = np.concatenate([
        np.zeros(len(X_hw_normal_s), dtype=int),
        np.ones( len(X_hw_attack_s), dtype=int),
    ])
    #assign label 0 to all normal sequences and label 1 to all attack sequences

    probs = predict_probs(model, X_cal)
    #run the model on the calibration set to get attack probabilities for each sequence

    best_thr    = 0.50
    #start with the default threshold of 0.5 as the fallback
    best_score  = float("inf")
    #track the best balance error (|FPR - FNR|) seen so far
    best_maxerr = float("inf")
    #track the best worst-case error max(FPR, FNR) seen so far for the fallback case

    for thr in np.arange(0.20, 0.81, 0.01):
        #sweep thresholds from 0.20 to 0.80 in steps of 0.01
        m = binary_metrics(y_cal, probs, thr=float(thr))
        #compute fpr and fnr at this threshold
        balance_err = abs(m["fpr"] - m["fnr"])
        #how balanced are fpr and fnr: 0 means perfectly balanced
        max_err     = max(m["fpr"], m["fnr"])
        #the worst of the two errors at this threshold

        if max_err <= 0.05:
            #if both fpr and fnr are under 5% then this is a safe threshold
            if balance_err < best_score:
                best_score = balance_err
                best_thr   = float(thr)
                #save this threshold if it has the best balance so far
        else:
            if max_err < best_maxerr:
                best_maxerr = max_err
                if best_score == float("inf"):
                    best_thr = float(thr)
                #use the fallback only if we haven't found a valid threshold yet

    print(f"\n[CALIBRATION] Calibrated threshold: {best_thr:.2f}")
    m_cal = binary_metrics(y_cal, probs, thr=best_thr)
    print(f"  FPR={m_cal['fpr']:.4f} | FNR={m_cal['fnr']:.4f} | "
          f"F1={m_cal['f1']:.4f} (on combined HW cal set)")
    #print the final chosen threshold and its metrics on the hardware calibration set
    return best_thr
    #return the best threshold found to be saved alongside the model weights


def prepare_mixed_data(scaler: StandardScaler):
    #this function builds the four-pool mixed fine-tuning dataset from all data sources
    print("[INFO] Loading original TRAIN split...")
    df_train = load_original_split_df(TRAIN_IDX)
    #load only the original training rows (using the saved indices to respect the split)

    print("[INFO] Loading original VAL split...")
    df_val = load_original_split_df(VAL_IDX)
    #load only the original validation rows

    print(f"[INFO] Loading hardware NORMAL CSV: {HW_NORMAL_CSV}")
    df_hw_normal = load_hardware_csv(HW_NORMAL_CSV, label=0)
    #load hardware normal traffic and label it as 0

    print(f"[INFO] Loading hardware ATTACK CSV: {HW_ATTACK_CSV}")
    df_hw_attack = load_hardware_csv(HW_ATTACK_CSV, label=1)
    #load hardware attack traffic and label it as 1

    print("[INFO] Building features...")
    df_train_feat, feature_cols = add_features(df_train)
    #build the 12 features for the training split
    df_val_feat,   _            = add_features(df_val)
    #build the 12 features for the validation split
    df_hw_normal_feat, _        = add_features(df_hw_normal)
    #build the 12 features for the hardware normal data
    df_hw_attack_feat, _        = add_features(df_hw_attack)
    #build the 12 features for the hardware attack data

    df_train_attack = df_train_feat[df_train_feat["label"] == 1].copy()
    #separate the original training data into attack-only rows for the attack pool
    df_train_normal = df_train_feat[df_train_feat["label"] == 0].copy()
    #separate the original training data into normal-only rows for the normal pool

    print("[INFO] Building sequences...")
    X_orig_attack,  y_orig_attack  = build_sequences_per_file(df_train_attack,     feature_cols)
    #build sliding window sequences from original attack rows
    X_orig_normal,  y_orig_normal  = build_sequences_per_file(df_train_normal,     feature_cols)
    #build sliding window sequences from original normal rows
    X_hw_normal,    y_hw_normal    = build_sequences_per_file(df_hw_normal_feat,   feature_cols)
    #build sliding window sequences from hardware normal rows
    X_hw_attack,    y_hw_attack    = build_sequences_per_file(df_hw_attack_feat,   feature_cols)
    #build sliding window sequences from hardware attack rows
    X_val,          y_val          = build_sequences_per_file(df_val_feat,         feature_cols)
    #build sliding window sequences from the validation split for evaluation during training

    X_orig_attack, y_orig_attack = subsample(X_orig_attack, y_orig_attack, MAX_ATTACK_SEQS)
    #cap the original attack pool to MAX_ATTACK_SEQS to control memory
    X_orig_normal, y_orig_normal = subsample(X_orig_normal, y_orig_normal, MAX_NORMAL_SEQS)
    #cap the original normal pool to MAX_NORMAL_SEQS
    X_hw_normal,   y_hw_normal   = subsample(X_hw_normal,   y_hw_normal,   MAX_HW_NORMAL_SEQS)
    #cap the hardware normal pool to MAX_HW_NORMAL_SEQS
    X_hw_attack,   y_hw_attack   = subsample(X_hw_attack,   y_hw_attack,   MAX_HW_ATTACK_SEQS)
    #cap the hardware attack pool to MAX_HW_ATTACK_SEQS

    print(f"\n[INFO] Pool sizes before mixing:")
    print(f"  HW normal    : {len(X_hw_normal):,} sequences  (label 0)")
    print(f"  HW attack    : {len(X_hw_attack):,} sequences  (label 1)  ← NEW")
    print(f"  Orig attack  : {len(X_orig_attack):,} sequences  (label 1)")
    print(f"  Orig normal  : {len(X_orig_normal):,} sequences  (label 0)")
    print(f"  Val set      : {len(X_val):,} sequences")
    #show how many sequences are in each pool before mixing so we can spot any empty or too-small pools

    from sklearn.preprocessing import StandardScaler as _SS
    CONT = [0, 1, 2]
    #column indices for the three continuous features that get scaled: Time, time_delta, Length

    X_for_refit = np.concatenate([
        X_orig_normal.reshape(-1, X_orig_normal.shape[-1])[:, CONT],
        X_orig_attack.reshape(-1, X_orig_attack.shape[-1])[:, CONT],
        X_hw_normal.reshape(-1,   X_hw_normal.shape[-1])[:, CONT],
        X_hw_attack.reshape(-1,   X_hw_attack.shape[-1])[:, CONT],
    ], axis=0)
    #combine continuous features from all four pools so the new scaler covers both lab-speed and hardware-speed traffic
    new_scaler = _SS()
    #create a fresh StandardScaler to be fitted on the combined data
    new_scaler.fit(X_for_refit)
    #fit the new scaler so its mean and std reflect both slow lab sessions and fast real hardware traffic
    scaler = new_scaler
    #replace the old scaler with the new one for the rest of this function
    print(f"[INFO] Scaler refitted on combined orig+hw data "
          f"({len(X_for_refit):,} samples)")
    print(f"  time_delta: mean={new_scaler.mean_[1]:.6f}s  "
          f"std={new_scaler.scale_[1]:.6f}s")
    print(f"  (old scaler mean was fitted on slow lab sessions only)")
    #show the new scaler statistics so we can verify the time_delta distribution is now realistic

    X_orig_attack_s = scale_sequences(X_orig_attack, scaler)
    #scale original attack sequences using the new refitted scaler
    X_orig_normal_s = scale_sequences(X_orig_normal, scaler)
    #scale original normal sequences
    X_hw_normal_s   = scale_sequences(X_hw_normal,   scaler)
    #scale hardware normal sequences
    X_hw_attack_s   = scale_sequences(X_hw_attack,   scaler)
    #scale hardware attack sequences
    X_val_s         = scale_sequences(X_val,         scaler)
    #scale validation sequences

    n_total = min(
        int(len(X_hw_normal_s)  / HW_NORMAL_FRAC),
        int(len(X_hw_attack_s)  / HW_ATTACK_FRAC),
        int(len(X_orig_attack_s)/ ORIG_ATTACK_FRAC),
        int(len(X_orig_normal_s)/ ORIG_NORMAL_FRAC),
    )
    #derive total mix size from the smallest pool so no pool needs to be oversampled

    n_hw_normal   = int(n_total * HW_NORMAL_FRAC)
    #how many sequences to take from the hardware normal pool based on the desired fraction
    n_hw_attack   = int(n_total * HW_ATTACK_FRAC)
    #how many sequences to take from the hardware attack pool
    n_orig_attack = int(n_total * ORIG_ATTACK_FRAC)
    #how many sequences to take from the original attack pool
    n_orig_normal = int(n_total * ORIG_NORMAL_FRAC)
    #how many sequences to take from the original normal pool

    idx_hw_n  = np.random.choice(len(X_hw_normal_s),   size=n_hw_normal,   replace=False)
    #randomly select n_hw_normal indices from the hardware normal pool without replacement
    idx_hw_a  = np.random.choice(len(X_hw_attack_s),   size=n_hw_attack,   replace=False)
    #randomly select n_hw_attack indices from the hardware attack pool
    idx_oa    = np.random.choice(len(X_orig_attack_s),  size=n_orig_attack, replace=False)
    #randomly select n_orig_attack indices from the original attack pool
    idx_on    = np.random.choice(len(X_orig_normal_s),  size=n_orig_normal, replace=False)
    #randomly select n_orig_normal indices from the original normal pool

    X_mix = np.concatenate([
        X_hw_normal_s[idx_hw_n],
        X_hw_attack_s[idx_hw_a],
        X_orig_attack_s[idx_oa],
        X_orig_normal_s[idx_on],
    ], axis=0)
    #stack all four selected pools into one mixed training array
    y_mix = np.concatenate([
        y_hw_normal[idx_hw_n],
        y_hw_attack[idx_hw_a],
        y_orig_attack[idx_oa],
        y_orig_normal[idx_on],
    ], axis=0)
    #stack the corresponding labels in the same order

    perm  = np.random.permutation(len(X_mix))
    #create a random permutation of all mixed indices so pools are not grouped together during training
    X_mix = X_mix[perm]
    #shuffle the mixed sequences
    y_mix = y_mix[perm]
    #shuffle the labels in the same order

    attack_pct = 100.0 * y_mix.sum() / len(y_mix)
    #compute what percentage of the final mixed set is labeled as attack
    print(f"\n[INFO] Mixed training set built:")
    print(f"  Total        : {len(X_mix):,}")
    print(f"  HW normal    : {n_hw_normal:,}   ({100*HW_NORMAL_FRAC:.0f}%)")
    print(f"  HW attack    : {n_hw_attack:,}   ({100*HW_ATTACK_FRAC:.0f}%)  ← NEW")
    print(f"  Orig attack  : {n_orig_attack:,}   ({100*ORIG_ATTACK_FRAC:.0f}%)")
    print(f"  Orig normal  : {n_orig_normal:,}   ({100*ORIG_NORMAL_FRAC:.0f}%)")
    print(f"  Attack label%: {attack_pct:.1f}%  (target ~{100*(HW_ATTACK_FRAC+ORIG_ATTACK_FRAC):.0f}%)")
    #show the final mix breakdown so we can verify the proportions look correct

    return (X_mix, y_mix.astype(np.float32),
            X_val_s, y_val.astype(np.float32),
            X_hw_normal_s, y_hw_normal.astype(np.float32),
            X_hw_attack_s, y_hw_attack.astype(np.float32),
            scaler)
    #return the mixed training set, validation set, hardware sets (for early stopping) and the refitted scaler


def fine_tune_one_model(det_name: str, cfg: Dict,
                        scaler: StandardScaler,
                        X_train, y_train,
                        X_val, y_val,
                        X_hw_normal, y_hw_normal,
                        X_hw_attack, y_hw_attack):
    #this function fine-tunes one detector model using the mixed data and saves the results

    print(f"\n{'='*70}")
    print(f"[INFO] Fine-tuning detector: {det_name}")
    print(f"{'='*70}")
    #print a separator so the output for each detector is easy to find in the logs

    model = build_model(cfg["type"]).to(DEVICE)
    #build the correct model architecture and move it to the right device
    model.load_state_dict(torch.load(cfg["ckpt_in"], map_location=DEVICE))
    #load the pre-trained weights from original training so we start from a good initialization
    print(f"[INFO] Loaded pre-trained weights from: {cfg['ckpt_in']}")
    print(f"[INFO] Using original scaler from      : {cfg['scaler_in']}")

    freeze_for_transfer(model, cfg["type"])
    #freeze the cnn layers and keep higher layers trainable so the model adapts without forgetting

    optimizer  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LR
    )
    #create an Adam optimizer that only updates the trainable (non-frozen) parameters at the lower fine-tuning lr
    criterion  = FocalLoss(alpha=0.5, gamma=2.0)
    #use focal loss instead of bce to focus training on hard ambiguous examples near the decision boundary

    best_score  = -float("inf")
    #track the best early-stopping score seen so far
    best_state  = None
    #will hold the model weights of the best epoch
    indices     = np.arange(len(X_train))
    #create an index array so we can shuffle the training data each epoch

    print(f"\n{'Ep':>3}  {'loss':>8}  {'val_f1':>7}  {'val_fpr':>8}  {'val_fnr':>8}  "
          f"{'hw_n_acc':>9}  {'hw_a_acc':>9}  {'score':>8}")
    print("-" * 80)
    #print the header row for the per-epoch training log

    for ep in range(1, FT_EPOCHS + 1):
        #run one epoch: shuffle data, go through all batches, update weights
        model.train()
        #set model to training mode so dropout is active
        np.random.shuffle(indices)
        #shuffle training data every epoch for better generalization

        total_loss = 0.0
        #accumulate loss across all batches so we can report average loss per epoch
        n_batches  = 0
        #count batches so we can divide total loss by number of batches

        for start in range(0, len(indices), FT_BATCH):
            #go through the training data batch by batch
            idx = indices[start:start + FT_BATCH]
            #get the indices for this batch
            xb  = torch.tensor(X_train[idx], dtype=torch.float32, device=DEVICE)
            #convert this batch to a tensor and move to the device
            yb  = torch.tensor(y_train[idx], dtype=torch.float32, device=DEVICE)
            #convert this batch's labels to a tensor

            optimizer.zero_grad()
            #clear the old gradients before computing new ones
            loss = criterion(model(xb), yb)
            #compute focal loss: how wrong is the model on this batch
            loss.backward()
            #compute gradients: how did each weight contribute to the error
            optimizer.step()
            #update weights: move them in the direction that reduces the error

            total_loss += float(loss.item())
            #add this batch's loss to the running total
            n_batches  += 1
            #count this batch

        val_prob = predict_probs(model, X_val)
        #run the model on the validation set to get attack probabilities
        val_m    = binary_metrics(y_val.astype(int), val_prob)
        #compute validation metrics with default threshold of 0.5

        hw_n_prob = predict_probs(model, X_hw_normal)
        #run the model on hardware normal sequences
        hw_n_acc  = float((hw_n_prob < 0.5).mean())
        #fraction of hardware normal sequences correctly predicted as normal (want this to be high)

        hw_a_prob = predict_probs(model, X_hw_attack)
        #run the model on hardware attack sequences
        hw_a_acc  = float((hw_a_prob >= 0.5).mean())
        #fraction of hardware attack sequences correctly predicted as attack (want this to be high)

        hw_fpr = 1.0 - hw_n_acc
        #hardware false positive rate: fraction of real normal traffic being wrongly flagged
        hw_fnr = 1.0 - hw_a_acc
        #hardware false negative rate: fraction of real attacks being missed

        score = val_m["f1"] - 1.5 * max(hw_fpr, hw_fnr)
        #composite early stopping score: reward high f1 but heavily penalize whichever of fpr/fnr is worse
        #this prevents the model from improving f1 while silently developing a bad imbalance

        print(f"{ep:>3}  {total_loss/max(n_batches,1):>8.4f}  "
              f"{val_m['f1']:>7.4f}  {val_m['fpr']:>8.4f}  {val_m['fnr']:>8.4f}  "
              f"{hw_n_acc:>9.4f}  {hw_a_acc:>9.4f}  {score:>8.4f}")
        #print one row of metrics for this epoch

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"     ↑ new best (score={score:.4f})")
        #if this epoch beat the best score so far then save its weights

    if best_state is None:
        raise RuntimeError("No best model state was captured.")
    #safety check: best_state should always be set by the first epoch at minimum

    model.load_state_dict(best_state)
    #reload the best weights found during training

    best_thr = calibrate_threshold(model, X_hw_normal, X_hw_attack)
    #sweep thresholds on hardware data to find the best decision boundary for this model

    val_prob      = predict_probs(model, X_val)
    #re-evaluate the validation set using the calibrated threshold
    val_m         = binary_metrics(y_val.astype(int), val_prob, thr=best_thr)
    #compute final validation metrics with the calibrated threshold
    hw_n_prob     = predict_probs(model, X_hw_normal)
    #re-evaluate hardware normal with the calibrated threshold
    hw_n_acc_cal  = float((hw_n_prob < best_thr).mean())
    #fraction of hardware normal correctly allowed after calibration
    hw_a_prob     = predict_probs(model, X_hw_attack)
    #re-evaluate hardware attack with the calibrated threshold
    hw_a_acc_cal  = float((hw_a_prob >= best_thr).mean())
    #fraction of hardware attack correctly detected after calibration

    print(f"\n[FINAL with thr={best_thr:.2f}]")
    print(f"  Val F1={val_m['f1']:.4f}  FPR={val_m['fpr']:.4f}  FNR={val_m['fnr']:.4f}")
    print(f"  HW normal acc={hw_n_acc_cal:.4f}  (target >0.90 — not over-blocking)")
    print(f"  HW attack acc={hw_a_acc_cal:.4f}  (target >0.85 — not over-permissive)")
    #print final calibrated results so we can immediately see if the model is safe for deployment

    torch.save(model.state_dict(), cfg["ckpt_out"])
    #save the fine-tuned model weights to disk
    with open(cfg["scaler_out"], "wb") as f:
        pickle.dump(scaler, f)
    #save the refitted scaler to disk so 4b and deployment use the same normalization
    with open(cfg["thresh_out"], "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thr, "detector": det_name}, f, indent=2)
    #save the calibrated threshold as json so 4b and deployment can load it without guessing

    with open(cfg["results_out"], "w", encoding="utf-8") as f:
        f.write(f"Detector: {det_name}\n")
        f.write(f"Mixture: {HW_NORMAL_FRAC:.0%} hw_normal / {HW_ATTACK_FRAC:.0%} hw_attack / "
                f"{ORIG_ATTACK_FRAC:.0%} orig_attack / {ORIG_NORMAL_FRAC:.0%} orig_normal\n")
        f.write(f"Loss: FocalLoss(alpha=0.5, gamma=2.0)\n")
        f.write(f"Best early-stop score : {best_score:.6f}\n\n")
        f.write(f"=== FINAL METRICS (threshold={best_thr:.2f}) ===\n")
        f.write(f"Val accuracy     : {val_m['acc']:.6f}\n")
        f.write(f"Val precision    : {val_m['prec']:.6f}\n")
        f.write(f"Val recall       : {val_m['rec']:.6f}\n")
        f.write(f"Val F1           : {val_m['f1']:.6f}\n")
        f.write(f"Val FPR          : {val_m['fpr']:.6f}\n")
        f.write(f"Val FNR          : {val_m['fnr']:.6f}\n")
        f.write(f"Val confusion matrix:\n{val_m['cm']}\n\n")
        f.write(f"HW normal acc    : {hw_n_acc_cal:.6f}  (fraction correctly allowed)\n")
        f.write(f"HW normal mean_p : {float(np.mean(hw_n_prob)):.6f}  (target < {best_thr:.2f})\n")
        f.write(f"HW attack acc    : {hw_a_acc_cal:.6f}  (fraction correctly detected)\n")
        f.write(f"HW attack mean_p : {float(np.mean(hw_a_prob)):.6f}  (target > {best_thr:.2f})\n")
    #write a full results report to a text file so we can review fine-tuning quality later

    print(f"[OK] Fine-tuned weights  -> {cfg['ckpt_out']}")
    print(f"[OK] Refitted scaler     -> {cfg['scaler_out']}  (refitted on orig+hw data)")
    print(f"[OK] Calibrated threshold -> {cfg['thresh_out']}")
    print(f"[OK] Results report      -> {cfg['results_out']}")
    #confirm all four output files were saved successfully

    print(f"\n[HEALTH CHECK]")
    print(f"  hw_normal_acc={hw_n_acc_cal:.4f}  (target >0.90)  "
          + ("OK" if hw_n_acc_cal > 0.90 else "WARNING: still over-blocking HW normal"))
    print(f"  hw_attack_acc={hw_a_acc_cal:.4f}  (target >0.85)  "
          + ("OK" if hw_a_acc_cal > 0.85 else "WARNING: still missing HW attacks"))
    #check if both hardware targets were met and print a clear ok or warning for each
    if hw_n_acc_cal <= 0.90 and hw_a_acc_cal <= 0.85:
        print("  BOTH thresholds failed — consider increasing FT_EPOCHS or adjusting mixture.")
    #if both failed suggest fixing the epochs or fractions
    elif hw_n_acc_cal <= 0.90:
        print("  SUGGESTION: increase HW_NORMAL_FRAC slightly (e.g. 0.40) and reduce ORIG_ATTACK_FRAC.")
    #if only the normal side failed suggest adding more hardware normal data
    elif hw_a_acc_cal <= 0.85:
        print("  SUGGESTION: increase HW_ATTACK_FRAC slightly (e.g. 0.30) and reduce ORIG_NORMAL_FRAC.")
    #if only the attack side failed suggest adding more hardware attack data


def main():
    random.seed(RANDOM_SEED)
    #seed python's random module for reproducibility
    np.random.seed(RANDOM_SEED)
    #seed numpy's random module for reproducibility
    torch.manual_seed(RANDOM_SEED)
    #seed pytorch's random module for reproducibility

    print("=" * 72)
    print(" NeuroGuard Transfer Learning  —  FT2 (HW Normal + HW Attack)")
    print("=" * 72)
    print(f"Device            : {DEVICE}")
    print(f"Dataset CSV       : {CSV_ALL}")
    print(f"HW Normal CSV     : {HW_NORMAL_CSV}")
    print(f"HW Attack CSV     : {HW_ATTACK_CSV}")
    print(f"Mixture           : {HW_NORMAL_FRAC:.0%} hw_normal | {HW_ATTACK_FRAC:.0%} hw_attack | "
          f"{ORIG_ATTACK_FRAC:.0%} orig_attack | {ORIG_NORMAL_FRAC:.0%} orig_normal")
    print(f"Loss              : FocalLoss(alpha=0.5, gamma=2.0)")
    print(f"Early stop score  : val_F1 - 1.5*max(hw_FPR, hw_FNR)")
    print()
    #print a startup summary so we can verify the configuration before waiting for training to finish

    for path in [CSV_ALL, HW_NORMAL_CSV, HW_ATTACK_CSV, TRAIN_IDX, VAL_IDX]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
    #safety check: verify all required input files exist before starting so we don't fail halfway through

    for det_name, cfg in DETECTOR_CONFIGS.items():
        #loop through all three detectors and fine-tune each one
        print(f"\n[INFO] Loading original scaler for {det_name} from: {cfg['scaler_in']}")
        with open(cfg["scaler_in"], "rb") as f:
            original_scaler = pickle.load(f)
        #load the original scaler that was fitted during the 3a/3b/3c training

        (X_train, y_train,
         X_val, y_val,
         X_hw_normal, y_hw_normal,
         X_hw_attack, y_hw_attack,
         refitted_scaler) = prepare_mixed_data(original_scaler)
        #build the four-pool mixed dataset using both hardware and original data

        fine_tune_one_model(
            det_name, cfg, refitted_scaler,
            X_train, y_train,
            X_val, y_val,
            X_hw_normal, y_hw_normal,
            X_hw_attack, y_hw_attack,
        )
        #fine-tune this detector and save all its outputs

    print("\n" + "=" * 72)
    print("DONE.")
    print("Next step: run 4b_retrain_mixed_ft2.py to retrain the RL policy.")
    print("=" * 72)
    #remind the user what to run next after fine-tuning completes


if __name__ == "__main__":
    main()
    #run the main function when this script is executed directly
