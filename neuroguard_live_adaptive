# neuroguard_live_adaptive.py
# ─────────────────────────────────────────────────────────────────────────────
# NeuroGuard Live — Adaptive Real-Time Deployment
#
# This is the no-gate deployment extended with an ONLINE RL that continues
# learning from every single decision it makes in production.
#
# HOW TO RUN (Raspberry Pi 5):
#   python neuroguard_live_adaptive.py --model cnn_only
#   python neuroguard_live_adaptive.py --model cnn_attn
#   python neuroguard_live_adaptive.py --model cnn_bilstm_attn
#
# ─── WHAT THE ONLINE RL ADDS ─────────────────────────────────────────────────
#
# The offline policy (loaded from rl_policy_best_ACCEPTED_*.pt) is a static
# snapshot trained on the original dataset plus hardware captures.  It cannot
# adapt to new attack patterns or traffic shifts that appear after training.
#
# The online RL layer wraps the offline policy and keeps learning:
#   1.  Every new 20-packet window produces a (state, action, reward, next_state)
#       transition that is pushed into a small live replay buffer.
#   2.  After every ONLINE_LEARN_EVERY decisions, a mini-batch is sampled from
#       that buffer and one gradient step is taken on the online Q-network.
#   3.  The online Q-network starts as an exact copy of the loaded offline policy
#       so it begins with all the knowledge that offline training produced.
#   4.  A target network is updated every ONLINE_TARGET_UPDATE steps for stable
#       Q-learning exactly as in rl_train.py.
#   5.  Because there is no ground-truth label at inference time, the reward
#       signal is derived from the detector probability and the action taken,
#       using the exact same reward logic as rl_env.py step().
#   6.  The online policy is saved to disk every ONLINE_SAVE_EVERY decisions so
#       progress survives a restart.
#
# ─── REWARD SIGNAL (mirrors rl_env.py step() exactly) ────────────────────────
#
# The detector probability (det_p) is used as a proxy for y_true:
#   det_p >= det_attack_thr  →  treat as attack  (y_true = 1)
#   det_p <  det_attack_thr  →  treat as normal  (y_true = 0)
#
# Rewards (matching rl_env.py values):
#   Normal traffic (det_p low):
#     ALLOW             → +2.5   (correct: traffic is safe, pass it through)
#     ALERT_ONLY / DEESCALATE → +0.8   (acceptable: advisory only, no disruption)
#     any block action  → -2.0   (penalty: blocking normal = false positive)
#     ESCALATE on normal→ -2.0   (extra penalty: don't escalate on safe traffic)
#     det_p very low + not ALLOW → additional -1.0 (the detector is confident it's normal)
#
#   Attack traffic (det_p high):
#     ALLOW             → -2.0   (penalty: letting an attack through = false negative)
#     NON_MITIGATE      → -0.5   (smaller penalty: at least some response)
#     real mitigation   → +1.5   (reward: attack correctly stopped)
#     mitigation + detector agrees → extra +0.5 (bonus: consistent with detector)
#     heavy action      → -0.1   (small cost: prefer lighter actions when possible)
#
#   Action cost (subtracted every step, matching rl_env.py action_costs):
#     Penalises heavier actions to encourage the agent to prefer minimal responses
#
# ─── ALGORITHM: Double DQN with experience replay ────────────────────────────
#
# Double DQN was chosen because:
#   - It directly matches the training algorithm used in rl_train.py so the
#     online updates are consistent with how the offline policy was learned.
#   - It reduces overestimation bias (a known problem with vanilla DQN) by
#     using the online net to select actions and the target net to evaluate them.
#   - It is stable and lightweight enough to run on a Raspberry Pi 5 in CPU mode
#     without impacting the 6ms inference latency of the detector.
#   - The small replay buffer (ONLINE_REPLAY_SIZE = 2000 transitions) keeps
#     memory usage under 5MB which is appropriate for the Pi.
#
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import collections
import json
import math
import os
import pickle
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Deque, Tuple
#importing all tools: argument parsing, deques, json, math, file paths, regex,
#signals, subprocess, threading and type hints

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#importing numpy for arrays and pytorch for the detector, offline rl, and online rl training


MODEL_DIR  = Path("/home/client2/neuroguard_models_3")
#the folder on the raspberry pi where all model weights, scalers and threshold files are stored
SSH_KEY    = "/home/client2/.ssh/id_rsa"
#path to the ssh private key used to connect to the broker raspberry pi
SSH_TARGET = "broker1@192.168.0.134"
#ssh username and ip address of the broker raspberry pi where tshark runs

DEFAULT_IFACE   = "wlan0"
#the wireless interface on the broker pi where tshark captures traffic
DISPLAY_REFRESH = 0.5
#how often the dashboard refreshes in seconds (every half second)

SEQ_LEN    = 20
#sliding window length: each decision is based on the last 20 packets exactly as in training
STEP       = 5
#a new decision fires after every 5 new packets arrive (same step size as training)
FEAT_DIM   = 12
#each packet has exactly 12 features: 3 continuous + 1 port flag + 6 tcp flags + 2 direction flags
STATE_DIM  = 16
#the rl agent's state vector always has 16 values
N_ACTIONS  = 16
#the rl agent can choose from 16 actions matching rl_env.py exactly
MQTT_PORTS = {1883, 8883}
#the two standard mqtt ports that we filter and monitor

THRESH_LOW  = 0.20
#below this detector probability we consider traffic normal for the dashboard display
THRESH_MID  = 0.70
#above this probability we consider traffic elevated (suspicious but not confirmed attack)
THRESH_HIGH = 0.95
#above this probability we consider traffic a confirmed high-threat attack

CONTINUOUS_IDX = [0, 1, 2]
#column indices for the three continuous features (Time, time_delta, Length) that get scaled by StandardScaler
#binary features in columns 3 to 11 are never scaled

# ─────────────────────────────────────────────────────────────────────────────
# ONLINE RL HYPERPARAMETERS
# These mirror rl_train.py as closely as possible while being lightweight
# enough to run in real time on a Raspberry Pi 5 CPU.
# ─────────────────────────────────────────────────────────────────────────────
ONLINE_REPLAY_SIZE    = 2000
#maximum number of transitions stored in the live replay buffer
#2000 is large enough to break short-term correlations between consecutive windows
#but small enough to keep memory usage under 5MB on the Pi
ONLINE_BATCH_SIZE     = 32
#number of transitions sampled per gradient update step
#32 is smaller than the 256 used in rl_train.py to keep update time under 1ms on CPU
ONLINE_LEARN_EVERY    = 5
#run one gradient update after every 5 new decisions
#this matches the STEP=5 cadence so the policy is updated after every new window batch
ONLINE_TARGET_UPDATE  = 50
#copy online net weights to the target net every 50 decisions
#matches the TARGET_UPDATE schedule in rl_train.py (scaled to the smaller update frequency)
ONLINE_LR             = 1e-4
#learning rate for the online Adam optimizer
#smaller than the 3e-4 used in offline training to prevent catastrophic forgetting
ONLINE_GAMMA          = 0.99
#discount factor for future rewards: same as rl_train.py
ONLINE_SAVE_EVERY     = 200
#save the online-adapted policy to disk every 200 decisions
#gives a checkpoint roughly every few minutes of live traffic
ONLINE_MIN_REPLAY     = 64
#do not start learning until at least this many transitions are in the replay buffer
#prevents learning from a tiny unrepresentative sample at the start

DET_NORMAL_LOW_THR    = 0.40
#below this probability we are very confident traffic is normal
#used in the reward function to add extra penalty for blocking clearly normal traffic
#matches det_normal_low_thr=0.40 from rl_env.py

A_ALLOW=0; A_RATE_LIMIT=1; A_TEMP_BLOCK=2; A_PERM_BLOCK=3
A_DROP_SYN=4; A_DROP_CONNECT=5; A_DELAY_CONNECT=6
A_LIMIT_PUBLISH=7; A_BLOCK_SUBSCRIBE=8; A_DISCONNECT=9
A_QUARANTINE=10; A_ISOLATE_NODE=11; A_REDUCE_QOS=12
A_ALERT_ONLY=13; A_ESCALATE=14; A_DEESCALATE=15
#action id constants matching rl_env.py exactly so the same actions mean the same things

ACTION_NAMES = {
    0:"ALLOW", 1:"RATE_LIMIT_IP", 2:"TEMP_BLOCK_IP", 3:"PERM_BLOCK_IP",
    4:"DROP_SYN_DELAY_TCP", 5:"DROP_CONNECT", 6:"DELAY_CONNECT",
    7:"LIMIT_PUBLISH", 8:"BLOCK_SUBSCRIBE", 9:"DISCONNECT_CLIENT",
    10:"QUARANTINE_CLIENT", 11:"ISOLATE_NODE", 12:"REDUCE_QOS",
    13:"ALERT_ONLY", 14:"ESCALATE", 15:"DEESCALATE",
}
#maps each action id to its human-readable name for display and logging

SAFE_ACTIONS  = {A_ALLOW, A_ALERT_ONLY, A_DEESCALATE}
#actions that are safe on normal traffic and do not count as false positives
HEAVY_ACTIONS = {A_TEMP_BLOCK, A_PERM_BLOCK, A_DISCONNECT, A_QUARANTINE, A_ISOLATE_NODE}
#heavy blocking actions that can break normal users so they need high confidence
NON_MITIGATE  = {A_ALLOW, A_ALERT_ONLY, A_ESCALATE, A_DEESCALATE}
#actions that do not directly stop an attack (matching rl_env.py NON_MITIGATE_ACTIONS)
LIGHT_ACTIONS = {A_RATE_LIMIT, A_DROP_SYN, A_DROP_CONNECT, A_DELAY_CONNECT,
                 A_LIMIT_PUBLISH, A_BLOCK_SUBSCRIBE, A_REDUCE_QOS}
#light mitigation actions that limit traffic without fully blocking it

ACTION_COSTS = {
    #per-action cost subtracted from reward every step — matching rl_env.py action_costs
    A_ALLOW: 0.0, A_ALERT_ONLY: 0.03, A_ESCALATE: 0.30, A_DEESCALATE: 0.05,
    A_RATE_LIMIT: 0.10, A_DROP_SYN: 0.15, A_DROP_CONNECT: 0.20,
    A_DELAY_CONNECT: 0.15, A_LIMIT_PUBLISH: 0.15, A_BLOCK_SUBSCRIBE: 0.15,
    A_REDUCE_QOS: 0.15, A_TEMP_BLOCK: 0.40, A_PERM_BLOCK: 0.70,
    A_DISCONNECT: 0.60, A_QUARANTINE: 0.70, A_ISOLATE_NODE: 0.90,
}
#heavier actions cost more so the agent prefers proportionate responses

RED="\033[91m"; GREEN="\033[92m"; YELLOW="\033[93m"
CYAN="\033[96m"; WHITE="\033[97m"; BOLD="\033[1m"; RESET="\033[0m"
#terminal colour escape codes for the live dashboard display

def colour_action(aid, name):
    #this function returns the action name wrapped in the appropriate terminal colour based on severity
    if aid == A_ALLOW:
        return f"{GREEN}{name}{RESET}"
    #ALLOW is green because it means traffic is safe and passes through
    if aid in {A_ALERT_ONLY, A_ESCALATE, A_DEESCALATE}:
        return f"{YELLOW}{name}{RESET}"
    #advisory actions are yellow because they are informational but not blocking
    if aid in HEAVY_ACTIONS:
        return f"{RED}{BOLD}{name}{RESET}"
    #heavy block actions are red and bold because they can disrupt normal users
    return f"{CYAN}{name}{RESET}"
    #light mitigations are cyan meaning they limit traffic without fully blocking it

def bar(v, w=20):
    #this function draws a filled unicode progress bar of width w representing value v between 0 and 1
    f = max(0, min(w, int(round(v * w))))
    #compute how many filled blocks fit based on the value
    s = "█"*f + "░"*(w-f)
    #build the bar string with filled and empty block characters
    if v >= 0.70: return f"{RED}{s}{RESET}"
    #high threat levels are shown in red
    if v >= 0.30: return f"{YELLOW}{s}{RESET}"
    #medium threat levels are shown in yellow
    return f"{GREEN}{s}{RESET}"
    #low threat levels are shown in green


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (identical to rl_env.py — do not change)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    #multi-head attention that focuses on the most important packets from multiple perspectives simultaneously
    def __init__(self, hidden_dim, num_heads=4):
        #hidden_dim is the size of each packet's feature vector and num_heads is how many attention patterns to learn
        super().__init__()
        assert hidden_dim % num_heads == 0
        #safety check: hidden_dim must divide evenly into num_heads
        self.num_heads = num_heads
        #store the number of heads
        self.head_dim  = hidden_dim // num_heads
        #each head processes this many features independently
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        #query projection: what is each packet looking for
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        #key projection: what information does each packet contain
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        #value projection: what information each packet provides when attended to
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        #final projection to combine all heads into one output vector
        self.scale = math.sqrt(self.head_dim)
        #scaling factor to prevent dot products from growing too large

    def forward(self, x):
        #x arrives as (batch, seq_len, hidden_dim)
        B, T, D = x.shape
        #unpack batch size, sequence length and feature dimension
        Q = self.q_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        #compute queries and reshape so each head handles its own slice
        K = self.k_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        #compute keys with the same reshape
        V = self.v_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        #compute values with the same reshape
        w = torch.softmax((Q @ K.transpose(-2,-1)) / self.scale, dim=-1)
        #compute attention weights: how much each packet should attend to every other packet
        return self.out_proj((w @ V).transpose(1,2).contiguous().view(B,T,D)).mean(dim=1)
        #apply weights to values, reassemble all heads, project and average across timesteps


class CNN_Only(nn.Module):
    #simple cnn-only detector that finds short-term local patterns in packet sequences
    def __init__(self, feat_dim=12, seq_len=20):
        #feat_dim is the number of features per packet and seq_len is the window size
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim,128,3,padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer looks at 3 packets at a time and learns 128 local patterns
            nn.Conv1d(128,128,3,padding=1),      nn.ReLU(), nn.BatchNorm1d(128))
            #second conv layer builds deeper representations from those 128 channels
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128*seq_len,128), nn.ReLU(),
            #flatten all features and compress to 128 with relu activation
            nn.Dropout(0.3), nn.Linear(128,1))
            #dropout to prevent overfitting then output one logit for attack vs normal
    def forward(self, x):
        return self.fc(self.conv(x.transpose(1,2))).squeeze(1)
        #transpose for conv1d then run cnn and fc layers and squeeze the output dimension


class CNN_Attention(nn.Module):
    #cnn combined with multi-head attention to focus on the most important packets in the window
    def __init__(self, feat_dim=12, num_heads=4):
        #feat_dim is the number of features per packet and num_heads is the number of attention heads
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim,128,3,padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer detects local patterns in groups of 3 packets
            nn.Conv1d(128,128,3,padding=1),      nn.ReLU(), nn.BatchNorm1d(128))
            #second conv layer learns richer representations
        self.attn = MultiHeadAttention(128, num_heads)
        #multi-head attention focuses on the most relevant steps in the cnn output
        self.fc   = nn.Sequential(
            nn.Linear(128,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
        #compress the attention output and make the final decision
    def forward(self, x):
        return self.fc(self.attn(self.conv(x.transpose(1,2)).transpose(1,2))).squeeze(1)
        #run cnn, transpose back for attention, then fc and squeeze


class CNN_BiLSTM_Attn(nn.Module):
    #the most powerful model: cnn for local patterns + bilstm for temporal context + attention for focusing
    def __init__(self, feat_dim=12, num_heads=4):
        #feat_dim is the number of features per packet and num_heads is the number of attention heads
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim,128,3,padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            #first conv layer finds burst patterns in consecutive packets
            nn.Conv1d(128,128,3,padding=1),      nn.ReLU(), nn.BatchNorm1d(128))
            #second conv layer goes deeper on those patterns
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        #bilstm reads the sequence both forward and backward; output size is 64*2=128
        self.attn   = MultiHeadAttention(128, num_heads)
        #attention focuses on the most informative timesteps in the bilstm output
        self.fc     = nn.Sequential(
            nn.Linear(128,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
        #compress and make the final attack vs normal decision
    def forward(self, x):
        h, _ = self.bilstm(self.conv(x.transpose(1,2)).transpose(1,2))
        #run cnn then bilstm to capture both local and sequential dependencies
        return self.fc(self.attn(h)).squeeze(1)
        #pass bilstm output through attention then fc and squeeze


class QNet(nn.Module):
    #the rl agent's q-network: takes a state vector and outputs one q-value per action
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        #state_dim is 16 and n_actions is 16 matching the training setup exactly
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,256), nn.ReLU(),
            #expand state from 16 to 256 neurons for capacity to learn complex policies
            nn.Linear(256,256),       nn.ReLU(),
            #second 256-neuron layer for deeper pattern recognition
            nn.Linear(256,n_actions))
        #output one q-value per action: agent picks the action with the highest q-value
    def forward(self, x):
        return self.net(x)
        #pass the 16-dimensional state through all three layers and return 16 q-values


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  — exact replica of add_features() from rl_env.py
# ─────────────────────────────────────────────────────────────────────────────

_PORT_RE  = re.compile(r'(\d+)\s*[>→]\s*(\d+)')
#pre-compiled regex to extract ports from both wireshark csv (→) and tshark output (>)
_SYN_RE   = re.compile(r'\bSYN\b')
#pre-compiled regex to detect the SYN flag using word boundaries so "SYNACK" won't match
_ACK_RE   = re.compile(r'\bACK\b')
#pre-compiled regex for the ACK flag
_FIN_RE   = re.compile(r'\bFIN\b')
#pre-compiled regex for the FIN flag
_RST_RE   = re.compile(r'\bRST\b')
#pre-compiled regex for the RST flag
_PSH_RE   = re.compile(r'\bPSH\b')
#pre-compiled regex for the PSH flag
_URG_RE   = re.compile(r'\bURG\b')
#pre-compiled regex for the URG flag

def extract_features_from_info(info_raw: str, length: float,
                                rel_time: float, time_delta: float) -> np.ndarray:
    #this function builds one 12-dimensional feature row from a tshark info string and packet metadata
    info_up = info_raw.upper()
    #convert the info string to uppercase for case-insensitive matching exactly as training did

    sport, dport = 0, 0
    #default port values in case the regex finds no match
    m = _PORT_RE.search(info_up)
    #search for a port pattern like "51781 > 1883" or "51781 → 1883" in the info string
    if m:
        try:
            sport = int(m.group(1))
            #extract source port from the first capture group
            dport = int(m.group(2))
            #extract destination port from the second capture group
        except ValueError:
            pass
        #if conversion fails keep the default zero values

    to_mqtt   = int(dport in MQTT_PORTS)
    #1 if the packet is going toward the mqtt broker (destination is mqtt port)
    from_mqtt = int(sport in MQTT_PORTS)
    #1 if the packet is coming from the mqtt broker (source is mqtt port)
    has_mqtt  = int(to_mqtt or from_mqtt)
    #1 if either source or destination is an mqtt port

    flag_syn = int(bool(_SYN_RE.search(info_up)))
    #1 if the SYN flag appears in the info string (connection initiation)
    flag_ack = int(bool(_ACK_RE.search(info_up)))
    #1 if the ACK flag appears (acknowledgement)
    flag_fin = int(bool(_FIN_RE.search(info_up)))
    #1 if the FIN flag appears (connection teardown)
    flag_rst = int(bool(_RST_RE.search(info_up)))
    #1 if the RST flag appears (reset, common in attacks)
    flag_psh = int(bool(_PSH_RE.search(info_up)))
    #1 if the PSH flag appears (push data immediately)
    flag_urg = int(bool(_URG_RE.search(info_up)))
    #1 if the URG flag appears (urgent pointer is significant)

    return np.array([
        rel_time,    #feature [0]: Time normalized to 0 at the first packet then growing in seconds
        time_delta,  #feature [1]: time difference from the previous packet (0 for the first)
        length,      #feature [2]: packet byte length
        has_mqtt,    #feature [3]: 1 if any mqtt port is involved
        flag_syn,    #feature [4]: SYN flag
        flag_ack,    #feature [5]: ACK flag
        flag_fin,    #feature [6]: FIN flag
        flag_rst,    #feature [7]: RST flag
        flag_psh,    #feature [8]: PSH flag
        flag_urg,    #feature [9]: URG flag
        to_mqtt,     #feature [10]: 1 if destination is mqtt port
        from_mqtt,   #feature [11]: 1 if source is mqtt port
    ], dtype=np.float32)
    #return as float32 numpy array in the exact same column order as training


def is_tcp_mqtt_port(info_raw: str) -> bool:
    #this function checks if a packet's info string contains an mqtt port reference
    info_up = info_raw.upper()
    #convert to uppercase for consistent matching
    return bool(re.search(r'\b1883\b|\b8883\b', info_up))
    #return True if the info field mentions port 1883 or 8883 with word boundaries (same filter as training)


def scale_sequence(seq: np.ndarray, scaler) -> np.ndarray:
    #this function applies the StandardScaler only to the three continuous features in the sequence
    s = seq.copy()
    #work on a copy so the original raw sequence is not modified
    s[:, CONTINUOUS_IDX] = scaler.transform(s[:, CONTINUOUS_IDX])
    #scale only Time, time_delta and Length columns (binary features in columns 3 to 11 are never scaled)
    return s
    #return the scaled sequence with the same shape as the input


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE REPLAY BUFFER
# Stores recent (state, action, reward, next_state, done) transitions so the
# online Q-network can learn from them in random mini-batches.  Thread-safe
# because the capture loop and training loop run in separate threads.
# ─────────────────────────────────────────────────────────────────────────────

class LiveReplayBuffer:
    #circular replay buffer that stores the most recent ONLINE_REPLAY_SIZE transitions
    def __init__(self, capacity: int):
        self._buf: Deque[Tuple] = collections.deque(maxlen=capacity)
        #a deque with a maximum length automatically drops the oldest entry when full
        self._lock = threading.Lock()
        #a lock to make push and sample thread-safe since both run in different threads

    def push(self, state, action, reward, next_state, done):
        #this function adds one transition to the buffer
        with self._lock:
            self._buf.append((
                np.array(state,      dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            ))
            #store everything as numpy arrays and scalars for fast tensor conversion later

    def sample(self, batch_size: int):
        #this function returns a random mini-batch of transitions as torch tensors
        with self._lock:
            if len(self._buf) < batch_size:
                return None
            #if the buffer doesn't have enough transitions yet return None
            indices = np.random.choice(len(self._buf), size=batch_size, replace=False)
            #choose batch_size unique indices at random (without replacement)
            batch = [self._buf[i] for i in indices]
            #fetch the selected transitions
        s   = torch.tensor(np.stack([t[0] for t in batch]), dtype=torch.float32)
        a   = torch.tensor([t[1] for t in batch],           dtype=torch.int64)
        r   = torch.tensor([t[2] for t in batch],           dtype=torch.float32)
        ns  = torch.tensor(np.stack([t[3] for t in batch]), dtype=torch.float32)
        d   = torch.tensor([t[4] for t in batch],           dtype=torch.float32)
        return s, a, r, ns, d
        #return states, actions, rewards, next_states and done flags as tensors

    def __len__(self):
        return len(self._buf)
        #return the current number of stored transitions


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE RL REWARD FUNCTION
# Mirrors rl_env.py step() reward logic exactly so the online updates are
# consistent with how the offline policy was trained.  The only difference is
# that det_p is used as a proxy for y_true since we have no ground-truth label.
# ─────────────────────────────────────────────────────────────────────────────

def compute_live_reward(action: int, det_p: float, det_attack_thr: float) -> float:
    #this function computes the reward for one decision using the detector probability as a label proxy
    reward = 0.0
    #start with zero reward and add or subtract based on the decision quality

    if det_p >= det_attack_thr:
        #detector says this is attack traffic (proxy for y_true == 1)
        if action == A_ALLOW:
            reward -= 2.0
            #allowing an attack through is a false negative — strong penalty
        elif action in NON_MITIGATE:
            reward -= 0.5
            #advisory actions during an attack are weak — smaller penalty
        else:
            reward += 1.5
            #real mitigation action during attack — give a reward
            if det_p >= THRESH_HIGH:
                reward += 0.5
            #extra bonus when the detector is very confident this is an attack
            if action in HEAVY_ACTIONS:
                reward -= 0.1
            #small cost for jumping straight to heavy blocking — prefer lighter responses first
    else:
        #detector says this is normal traffic (proxy for y_true == 0)
        if action == A_ALLOW:
            reward += 2.5
            #correct: letting safe traffic through — biggest positive reward
        elif action in SAFE_ACTIONS:
            reward += 0.8
            #ALERT_ONLY or DEESCALATE on normal are acceptable — small positive reward
        else:
            reward -= 2.0
            #blocking normal traffic is a false positive — strong penalty
        if det_p < DET_NORMAL_LOW_THR and action != A_ALLOW:
            reward -= 1.0
            #detector is very confident it's normal and agent still blocked it — extra penalty
        if action == A_ESCALATE:
            reward -= 2.0
            #explicitly penalize ESCALATE on normal traffic to prevent escalation spam

    reward -= float(ACTION_COSTS.get(action, 0.0))
    #subtract the action's cost so the agent prefers proportionate responses matching rl_env.py
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# LIVE STATE TRACKER  — mirrors rl_env.py internal state tracking exactly
# ─────────────────────────────────────────────────────────────────────────────

class LiveStateTracker:
    #this class tracks the rl agent's internal context (previous action, counters, escalation level) across decisions
    def __init__(self):
        self.prev_action      = A_ALLOW
        #start assuming the last action was ALLOW which is the baseline safe state
        self.escalation_level = 0
        #start with no escalation (level 0 = no active threat response)
        self.fp_counter       = 0
        #counts how often the agent has been blocking what might be normal traffic
        self.fn_counter       = 0
        #counts how often the agent has been missing what looks like an attack

    def build_state(self, seq_scaled: np.ndarray, det_p: float) -> np.ndarray:
        #this function builds the 16-dimensional state vector from the scaled sequence and current context
        extras = np.array([
            det_p,
            #[0] detector attack probability for this window (0 = very likely normal, 1 = very likely attack)
            float(np.mean(seq_scaled[:, 1])),
            #[1] mean inter-packet time across all 20 packets in the window (scaled)
            float(np.mean(seq_scaled[:, 2])),
            #[2] mean packet size across all 20 packets in the window (scaled)
            float(np.mean(seq_scaled[:, 3])),
            #[3] fraction of packets in the window that involve an mqtt port
            float(self.prev_action)      / 15.0,
            #[4] the last action taken normalized to 0-1 so the agent knows its recent history
            float(self.escalation_level) / 5.0,
            #[5] current escalation level normalized to 0-1 (maximum is 5)
            float(self.fp_counter)       / 10.0,
            #[6] false positive pressure normalized to 0-1 (maximum is 10)
            float(self.fn_counter)       / 10.0,
            #[7] false negative pressure normalized to 0-1 (maximum is 10)
        ], dtype=np.float32)
        flags_mean = np.mean(seq_scaled[:, 4:12], axis=0).astype(np.float32)
        #[8-15] mean of each of the 8 flag and direction features across the 20-packet window
        return np.concatenate([extras, flags_mean]).astype(np.float32)
        #combine the 8 scalar features and the 8 flag means to get the 16-dimensional state vector

    def update(self, action: int, det_p: float, det_attack_thr: float):
        #this function updates the counters and escalation level after each decision
        if action == A_ESCALATE:
            self.escalation_level = min(self.escalation_level + 1, 5)
            #escalate means increase the response severity (cap at 5)
        elif action == A_DEESCALATE:
            self.escalation_level = max(self.escalation_level - 1, 0)
            #deescalate means decrease the response severity (floor at 0)

        if det_p >= det_attack_thr:
            #detector says this looks like an attack
            if action in NON_MITIGATE:
                self.fn_counter = min(self.fn_counter + 1, 10)
                #if the agent didn't mitigate it then increase fn pressure (we might be missing attacks)
            else:
                self.fn_counter = max(self.fn_counter - 1, 0)
                #if the agent mitigated it then decrease fn pressure
        else:
            #detector says this looks like normal traffic
            if action == A_ALLOW:
                self.fp_counter = max(self.fp_counter - 2, 0)
                #agent correctly allowed normal traffic so decrease fp pressure
            elif action not in NON_MITIGATE:
                self.fp_counter = min(self.fp_counter + 2, 10)
                #agent blocked normal-looking traffic so increase fp pressure

        self.prev_action = action
        #remember the action that was just taken for the next state vector


# ─────────────────────────────────────────────────────────────────────────────
# LIVE STATS
# ─────────────────────────────────────────────────────────────────────────────

class LiveStats:
    #this class collects and stores live statistics for the dashboard display in a thread-safe way
    def __init__(self):
        self._lock             = threading.Lock()
        #a lock to make all stat updates thread-safe since the capture loop and dashboard run in separate threads
        self.total_packets     = 0
        #total number of packets seen since the system started
        self.total_decisions   = 0
        #total number of decisions made (one per STEP new packets when buffer is full)
        self.attack_decisions  = 0
        #decisions where the detector probability was above THRESH_HIGH (confirmed high-threat)
        self.normal_decisions  = 0
        #decisions where the action is ALLOW (traffic confirmed safe, either by model or gate)
        self.action_counts     = collections.Counter()
        #counts how often each action name has been chosen across all decisions
        self.recent_detp       = collections.deque(maxlen=50)
        #stores the last 50 detector probabilities for computing the recent average threat level
        self.recent_actions    = collections.deque(maxlen=10)
        #stores the last 10 (action_id, action_name, det_p) tuples for the dashboard display
        self.recent_src_ips    = collections.deque(maxlen=5)
        #stores the last 5 source ip addresses seen for the dashboard
        self.start_time        = time.time()
        #records when the system started so we can compute uptime
        self.response_times_ms = collections.deque(maxlen=200)
        #stores the last 200 decision latencies in milliseconds for response time statistics
        self.total_reward      = 0.0
        #running total of all live rewards so we can track the agent's performance over time
        self.online_updates    = 0
        #counts how many online gradient updates have been performed
        self.replay_size       = 0
        #current number of transitions stored in the live replay buffer

    def record_packet(self, src_ip: str):
        #this function records that one new packet was seen
        with self._lock:
            self.total_packets += 1
            #increment the total packet counter
            self.recent_src_ips.append(src_ip)
            #add the source ip to the recent ip deque

    def record_decision(self, det_p: float, aid: int, aname: str, ms: float,
                        reward: float):
        #this function records the result of one decision including the live reward
        with self._lock:
            self.total_decisions += 1
            #increment the total decision counter
            self.recent_detp.append(det_p)
            #add this decision's detector probability to the recent threat level deque
            self.action_counts[aname] += 1
            #count this action name
            self.recent_actions.append((aid, aname, det_p, reward))
            #save the full action record including reward for the dashboard
            self.response_times_ms.append(ms)
            #save the latency for response time statistics
            self.total_reward += reward
            #accumulate the reward for the average reward display
            if aid == A_ALLOW:
                self.normal_decisions += 1
            #ALLOW means traffic is safe — count as normal regardless of det_p
            elif det_p >= THRESH_HIGH:
                self.attack_decisions += 1
            #very high det_p means confirmed attack
            elif det_p < THRESH_LOW:
                self.normal_decisions += 1
            #very low det_p also counts as normal

    def record_online_update(self, replay_size: int):
        #this function records that one online gradient update was performed
        with self._lock:
            self.online_updates += 1
            #increment the online update counter
            self.replay_size = replay_size
            #update the current replay buffer size for the dashboard

    def snapshot(self) -> dict:
        #this function returns a thread-safe copy of all stats at this moment in time
        with self._lock:
            rt = list(self.response_times_ms)
            #copy the response times list while holding the lock
            td = max(self.total_decisions, 1)
            #avoid division by zero when computing the average reward
            return dict(
                total_packets    = self.total_packets,
                total_decisions  = self.total_decisions,
                attack_decisions = self.attack_decisions,
                normal_decisions = self.normal_decisions,
                action_counts    = dict(self.action_counts),
                recent_detp      = list(self.recent_detp),
                recent_actions   = list(self.recent_actions),
                recent_src_ips   = list(self.recent_src_ips),
                uptime_s         = time.time() - self.start_time,
                avg_ms  = float(np.mean(rt))           if rt else 0.0,
                #average decision latency across the last 200 decisions
                min_ms  = float(np.min(rt))            if rt else 0.0,
                #minimum decision latency seen
                max_ms  = float(np.max(rt))            if rt else 0.0,
                #maximum decision latency seen
                p95_ms  = float(np.percentile(rt, 95)) if rt else 0.0,
                #95th percentile latency: 95% of decisions were faster than this
                avg_reward       = self.total_reward / td,
                #average reward per decision: positive means agent is performing well
                online_updates   = self.online_updates,
                #total number of gradient updates performed so far
                replay_size      = self.replay_size,
                #current number of transitions in the live replay buffer
            )


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(stats: dict, model_name: str, thr: float):
    #this function clears the terminal and prints a live dashboard of all key metrics
    os.system("clear")
    #clear the terminal for a clean refresh
    up = stats["uptime_s"]
    h, r = divmod(int(up), 3600); m, s = divmod(r, 60)
    #convert uptime seconds to hours, minutes and seconds for display
    pr  = stats["total_packets"]   / max(up, 1)
    #packets per second since the system started
    dr  = stats["total_decisions"] / max(up, 1)
    #decisions per second since the system started
    dl  = stats["recent_detp"]
    #list of recent detector probabilities
    avg = float(np.mean(dl)) if dl else 0.0
    #average detector probability over the last 50 decisions

    if avg >= THRESH_HIGH:  tl = f"{RED}{BOLD}! HIGH THREAT  !{RESET}"
    elif avg >= THRESH_MID: tl = f"{YELLOW} ELEVATED{RESET}"
    elif avg >= THRESH_LOW: tl = f"{YELLOW}~ SUSPICIOUS{RESET}"
    else:                   tl = f"{GREEN}  NORMAL{RESET}"
    #choose the threat level label and colour based on the average detector probability

    W = 60
    print(f"{BOLD}{'═'*W}{RESET}")
    print(f"  {CYAN}{BOLD}NeuroGuard Adaptive  |  Model: {model_name}{RESET}")
    print(f"  Uptime: {h:02d}:{m:02d}:{s:02d}  │  thr={thr:.2f}  │  Ctrl+C to stop")
    print(f"{'═'*W}")
    #print the header section showing uptime and calibrated threshold
    print(f"  Packets captured : {WHITE}{stats['total_packets']:,}{RESET}  ({pr:.1f} pkt/s)")
    #show total packets and current packet rate
    print(f"  Decisions made   : {WHITE}{stats['total_decisions']:,}{RESET}  ({dr:.1f} dec/s)")
    #show total decisions and current decision rate
    print(f"\n  Threat level  {bar(avg)}  {avg:.2f}  {tl}\n")
    #show the visual threat bar with the average probability and label
    print(f"  {BOLD}Response Time (per 20-packet window → 1 decision):{RESET}")
    print(f"  Avg:{WHITE}{stats['avg_ms']:.3f} ms{RESET}  "
          f"Min:{GREEN}{stats['min_ms']:.3f} ms{RESET}  "
          f"Max:{YELLOW}{stats['max_ms']:.3f} ms{RESET}  "
          f"p95:{WHITE}{stats['p95_ms']:.3f} ms{RESET}")
    #show response time statistics: how fast is the system making decisions
    if stats["avg_ms"] > 0:
        print(f"  Throughput: {WHITE}{1000.0/stats['avg_ms']:.1f} dec/s{RESET}")
    #show throughput in decisions per second derived from average latency
    print(f"\n  {BOLD}Online RL status:{RESET}")
    avg_r = stats["avg_reward"]
    r_col = GREEN if avg_r >= 0 else RED
    print(f"  Updates: {WHITE}{stats['online_updates']:,}{RESET}  "
          f"Replay: {WHITE}{stats['replay_size']}/{ONLINE_REPLAY_SIZE}{RESET}  "
          f"Avg reward: {r_col}{avg_r:+.3f}{RESET}")
    #show how many online gradient updates have run, the replay buffer fill level
    #and the average reward (green if positive, red if negative)
    print(f"\n  {BOLD}Last decisions:{RESET}")
    for entry in reversed(list(stats["recent_actions"])):
        aid, an, dp, rw = entry[0], entry[1], entry[2], entry[3]
        rw_col = GREEN if rw >= 0 else RED
        print(f"    det_p={dp:.2f}  r={rw_col}{rw:+.1f}{RESET}  →  {colour_action(aid, an)}")
    #print the last few decisions in reverse order with detector probability, reward and action colour
    print(f"\n  {BOLD}Top actions (all time):{RESET}")
    td = max(stats["total_decisions"], 1)
    for an, cnt in sorted(stats["action_counts"].items(), key=lambda x: -x[1])[:5]:
        aid = next((k for k, v in ACTION_NAMES.items() if v == an), 0)
        print(f"    {colour_action(aid, f'{an:25s}')}  {cnt:6d}  ({100.0*cnt/td:5.1f}%)")
    #show the top 5 actions the agent has taken sorted by how often they were chosen
    if stats["recent_src_ips"]:
        print(f"\n  {BOLD}Recent source IPs:{RESET}")
        for ip in list(dict.fromkeys(stats["recent_src_ips"]))[-5:]:
            print(f"    {WHITE}{ip}{RESET}")
    #show the most recent unique source ip addresses seen in the captured traffic
    print(f"\n{BOLD}{'═'*W}{RESET}")
    unc = (stats["total_decisions"]
           - stats["attack_decisions"]
           - stats["normal_decisions"])
    print(f"  {RED}High-threat={stats['attack_decisions']}{RESET}  "
          f"{GREEN}Normal={stats['normal_decisions']}{RESET}  "
          f"Uncertain={unc}")
    print(f"{'═'*W}")
    #show the breakdown of decision outcomes: confirmed attacks, normal and uncertain


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class NeuroGuardAdaptive:
    #main class that loads all models and runs the adaptive real-time detection loop

    def __init__(self, model_name: str, iface: str):
        #model_name is the detector to use (cnn_only, cnn_attn or cnn_bilstm_attn)
        #iface is the network interface name shown in the dashboard
        self.model_name = model_name
        #store the model name for display and file lookup
        self.iface      = iface
        #store the interface name
        self.device     = "cpu"
        #raspberry pi 5 has no cuda so we always use cpu
        self.running    = False
        #flag that controls the main capture loop

        # ── load calibrated threshold ──────────────────────────────────────
        self.det_attack_thr = 0.50
        #default decision threshold before we try to load the calibrated one from 4a
        for suffix in ("_ft2", "_ft"):
            tmap = {
                "cnn_only":        f"threshold_cnn_only{suffix}.json",
                "cnn_attn":        f"threshold_cnn_attention{suffix}.json",
                "cnn_bilstm_attn": f"threshold_cnn_bilstm_attn{suffix}.json",
            }
            tp = MODEL_DIR / tmap[model_name]
            if tp.exists():
                with open(tp) as f:
                    self.det_attack_thr = float(json.load(f).get("threshold", 0.50))
                print(f"[INFO] Calibrated threshold: {self.det_attack_thr:.2f}  ({tp.name})")
                break
        #try to load the calibrated threshold from 4a (prefer ft2 over ft)
        else:
            print("[WARNING] No threshold JSON found. Using default 0.50.")
            print("          Run 4a_ft2_fixed.py first.")
            #warn if no calibrated threshold is found so the user knows to run 4a first

        # ── load scaler ───────────────────────────────────────────────────
        for suffix in ("_ft2", "_ft"):
            smap = {
                "cnn_only":        f"scaler_cnn_only{suffix}.pkl",
                "cnn_attn":        f"scaler_cnn_attention{suffix}.pkl",
                "cnn_bilstm_attn": f"scaler_cnn_bilstm_attn{suffix}.pkl",
            }
            sp = MODEL_DIR / smap[model_name]
            if sp.exists(): break
        #find the most recent scaler file (prefer ft2 over ft)
        print(f"[INFO] Loading scaler from {sp.name}")
        with open(sp, "rb") as f:
            self.scaler = pickle.load(f)
        #load the scaler that was fitted during fine-tuning for live feature normalization

        # ── load detector (frozen — never updated online) ─────────────────
        for suffix in ("_ft2", "_ft"):
            cmap = {
                "cnn_only":        f"detector_cnn_only{suffix}.pt",
                "cnn_attn":        f"detector_cnn_attention{suffix}.pt",
                "cnn_bilstm_attn": f"detector_cnn_bilstm_attn{suffix}.pt",
            }
            cp = MODEL_DIR / cmap[model_name]
            if cp.exists(): break
        #find the most recent fine-tuned detector weights (prefer ft2 over ft)
        print(f"[INFO] Loading detector from {cp.name}")
        if model_name == "cnn_only":
            self.detector = CNN_Only(FEAT_DIM, SEQ_LEN)
            #build the cnn-only detector architecture
        elif model_name == "cnn_attn":
            self.detector = CNN_Attention(FEAT_DIM, 4)
            #build the cnn + attention detector architecture
        else:
            self.detector = CNN_BiLSTM_Attn(FEAT_DIM, 4)
            #build the full cnn + bilstm + attention detector architecture
        self.detector.load_state_dict(torch.load(cp, map_location=self.device))
        #load the fine-tuned weights into the detector
        self.detector.eval()
        #set to evaluation mode so dropout is disabled and predictions are deterministic
        for p in self.detector.parameters():
            p.requires_grad = False
        #freeze all detector parameters — the detector is never updated online
        #only the RL policy adapts; the detector stays fixed

        # ── load offline RL policy into the online Q-network ─────────────
        for suffix in ("_ft2", "_ft"):
            rmap = {
                "cnn_only":        f"rl_policy_best_ACCEPTED_cnn_only{suffix}.pt",
                "cnn_attn":        f"rl_policy_best_ACCEPTED_cnn_attn{suffix}.pt",
                "cnn_bilstm_attn": f"rl_policy_best_ACCEPTED_cnn_bilstm_attn{suffix}.pt",
            }
            rp = MODEL_DIR / rmap[model_name]
            if rp.exists(): break
        #find the most recent accepted rl policy (prefer ft2 over ft)

        # check if a previously saved adaptive policy exists and is newer
        adaptive_path = MODEL_DIR / f"rl_policy_adaptive_{model_name}.pt"
        if adaptive_path.exists():
            rp = adaptive_path
            print(f"[INFO] Resuming from adaptive policy: {rp.name}")
            #if we have previously saved an online-adapted version, continue from there
        else:
            print(f"[INFO] Starting online RL from offline policy: {rp.name}")
            #otherwise start from the offline policy so we keep all training knowledge

        self.q_net = QNet(STATE_DIM, N_ACTIONS)
        #build the online q-network architecture
        self.q_net.load_state_dict(torch.load(rp, map_location=self.device))
        #load the policy weights — either adaptive (resumed) or offline (fresh start)
        self.q_net.train()
        #set to training mode so the online updates can update batch-norm statistics

        self.target_net = QNet(STATE_DIM, N_ACTIONS)
        #build the target q-network with the same architecture
        self.target_net.load_state_dict(self.q_net.state_dict())
        #initialize the target network as an exact copy of the online network
        self.target_net.eval()
        #the target network is always in evaluation mode — it is only updated by hard copy
        for p in self.target_net.parameters():
            p.requires_grad = False
        #freeze the target network so gradients never flow through it during updates

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=ONLINE_LR)
        #Adam optimizer for the online Q-network updates at the smaller learning rate
        self.loss_fn   = nn.SmoothL1Loss()
        #Smooth L1 loss (Huber loss) matching rl_train.py for stable Q-learning
        self.replay    = LiveReplayBuffer(ONLINE_REPLAY_SIZE)
        #the live replay buffer that stores recent transitions from production traffic

        # ── runtime state ─────────────────────────────────────────────────
        self.packet_buffer               = collections.deque(maxlen=SEQ_LEN)
        #a circular buffer that holds the last SEQ_LEN raw (unscaled) feature rows
        self.packets_since_last_decision = 0
        #counts how many new packets have arrived since the last decision was made
        self.capture_start_time: Optional[float] = None
        #the timestamp of the very first mqtt packet seen, used to normalize all subsequent times
        self.prev_rel_time: float                = 0.0
        #the relative time of the previous packet, used to compute time_delta
        self.prev_state: Optional[np.ndarray]    = None
        #the state vector from the previous decision, needed to form (s, a, r, s') transitions
        self.prev_action: Optional[int]          = None
        #the action taken in the previous decision, needed for the same reason
        self.decision_count = 0
        #counts total decisions made, used to trigger learning and target network updates

        self.state_tracker = LiveStateTracker()
        #tracks the rl agent's context (counters, escalation, previous action) across decisions
        self.stats         = LiveStats()
        #collects live statistics for the dashboard display
        self._train_lock   = threading.Lock()
        #a lock so the training thread and the capture thread don't conflict on the q_net weights

        self.adaptive_path = adaptive_path
        #path where the adapted policy will be saved periodically

        print(f"[INFO] Ready. det_attack_thr={self.det_attack_thr:.2f}")
        print(f"[INFO] Online RL: lr={ONLINE_LR} gamma={ONLINE_GAMMA} "
              f"learn_every={ONLINE_LEARN_EVERY} target_update={ONLINE_TARGET_UPDATE}")
        #confirm the system is ready and print the online RL configuration

    # ── detector inference ────────────────────────────────────────────────────

    def _run_detector(self, seq_raw: np.ndarray) -> float:
        #this function scales the raw sequence and runs the detector to get an attack probability
        seq_scaled = scale_sequence(seq_raw, self.scaler)
        #scale the continuous features using the fitted scaler
        x = torch.tensor(seq_scaled[None, ...], dtype=torch.float32)
        #add a batch dimension and convert to a torch tensor
        with torch.no_grad():
            return float(torch.sigmoid(self.detector(x)).item())
        #run the detector without computing gradients and convert the logit to a probability

    # ── online RL action selection ────────────────────────────────────────────

    def _select_action(self, state: np.ndarray) -> int:
        #this function picks the action with the highest q-value from the online network
        with self._train_lock:
            #hold the lock to prevent a concurrent training update from changing weights mid-inference
            self.q_net.eval()
            #switch to eval mode briefly for inference
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            #add a batch dimension and convert to tensor
            with torch.no_grad():
                action = int(torch.argmax(self.q_net(s_t), dim=1).item())
            #run the q-network and pick the action with the highest q-value
            self.q_net.train()
            #switch back to train mode so online updates can proceed
        return action

    # ── online Double DQN update ──────────────────────────────────────────────

    def _online_update(self):
        #this function performs one Double DQN gradient step on the online Q-network
        batch = self.replay.sample(ONLINE_BATCH_SIZE)
        if batch is None:
            return
        #if the replay buffer doesn't have enough transitions yet skip this update

        states, actions, rewards, next_states, dones = batch
        #unpack the sampled mini-batch

        with self._train_lock:
            #hold the lock so inference and training don't conflict on the q_net
            self.q_net.train()
            #ensure we are in training mode for this update

            # Double DQN: use online net to SELECT the best next action
            # and target net to EVALUATE its Q-value
            # This reduces overestimation bias compared to vanilla DQN
            # and exactly matches the Double DQN logic in rl_train.py
            with torch.no_grad():
                next_actions   = torch.argmax(self.q_net(next_states), dim=1, keepdim=True)
                #online network selects the best action in the next state
                next_q_target  = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                #target network evaluates how good that action is
                target_q       = rewards + ONLINE_GAMMA * next_q_target * (1.0 - dones)
                #Bellman target: immediate reward plus discounted future value (zero if done)

            current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            #Q-value the online network currently assigns to the action that was actually taken

            loss = self.loss_fn(current_q, target_q)
            #Smooth L1 loss between current estimate and Bellman target

            self.optimizer.zero_grad()
            #clear gradients from the previous update step
            loss.backward()
            #compute gradients via backpropagation
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
            #clip gradients to prevent exploding gradients matching rl_train.py
            self.optimizer.step()
            #apply the gradient update to the online Q-network weights

        self.stats.record_online_update(len(self.replay))
        #record that one update happened and report the current replay buffer size

    # ── target network update ─────────────────────────────────────────────────

    def _update_target_net(self):
        #this function copies the online Q-network weights into the target network
        with self._train_lock:
            self.target_net.load_state_dict(self.q_net.state_dict())
        #hard copy of all weights — matching TARGET_UPDATE logic in rl_train.py

    # ── save adapted policy ───────────────────────────────────────────────────

    def _save_adaptive_policy(self):
        #this function saves the current online Q-network weights to disk
        with self._train_lock:
            torch.save(self.q_net.state_dict(), self.adaptive_path)
        #atomic save under the lock so we never write a half-updated file
        print(f"[INFO] Adaptive policy saved → {self.adaptive_path.name}  "
              f"(decisions={self.decision_count})")

    # ── main decision window ──────────────────────────────────────────────────

    def _process_window(self):
        #this function is called when SEQ_LEN packets are in the buffer and STEP new ones have arrived
        seq_raw    = np.array(list(self.packet_buffer), dtype=np.float32)
        #convert the deque of 20 feature rows to a numpy array
        t0         = time.perf_counter()
        #start timing the decision so we can measure latency

        det_p      = self._run_detector(seq_raw)
        #run the detector to get the attack probability for this 20-packet window
        seq_scaled = scale_sequence(seq_raw, self.scaler)
        #scale the raw sequence for the rl state builder
        state      = self.state_tracker.build_state(seq_scaled, det_p)
        #build the 16-dimensional state vector from the scaled sequence and current agent context
        action     = self._select_action(state)
        #pick the best action from the online Q-network
        ms         = (time.perf_counter() - t0) * 1000.0
        #compute how long the full decision took in milliseconds

        reward = compute_live_reward(action, det_p, self.det_attack_thr)
        #compute the reward for this decision using the detector probability as a label proxy

        # store a complete (s, a, r, s') transition in the replay buffer
        # we use the current state as next_state for the previous transition
        # (the state we just built is the 'next state' that followed the previous action)
        if self.prev_state is not None and self.prev_action is not None:
            self.replay.push(
                self.prev_state,
                self.prev_action,
                reward,
                state,
                0.0,
                #done=0 always in live deployment since there are no episode boundaries
            )
            #push the complete transition now that we have the next state

        # update the online Q-network and target network on schedule
        self.decision_count += 1
        #count this decision

        if (self.decision_count % ONLINE_LEARN_EVERY == 0
                and len(self.replay) >= ONLINE_MIN_REPLAY):
            self._online_update()
            #run one gradient step every ONLINE_LEARN_EVERY decisions if the buffer is ready

        if self.decision_count % ONLINE_TARGET_UPDATE == 0:
            self._update_target_net()
            #hard-copy weights to target net every ONLINE_TARGET_UPDATE decisions

        if self.decision_count % ONLINE_SAVE_EVERY == 0:
            self._save_adaptive_policy()
            #save the adapted policy to disk every ONLINE_SAVE_EVERY decisions

        aname = ACTION_NAMES.get(action, "UNKNOWN")
        #look up the human-readable action name
        self.state_tracker.update(action, det_p, self.det_attack_thr)
        #update the agent's internal context based on the action just taken
        self.stats.record_decision(det_p, action, aname, ms, reward)
        #record the decision, latency and reward in the stats for the dashboard

        self.prev_state  = state
        #save this state so the next decision can form the (s, a, r, s') transition
        self.prev_action = action
        #save this action for the same reason

    # ── packet ingestion ──────────────────────────────────────────────────────

    def _on_packet(self, ts: float, src_ip: str, info_raw: str, length: float):
        #this function processes one packet that has passed the tcp+mqtt-port filter
        self.stats.record_packet(src_ip)
        #count this packet and record its source ip

        if self.capture_start_time is None:
            self.capture_start_time = ts
            #the very first packet sets the capture start time so all times are relative to it
            rel_time   = 0.0
            #the first packet is at time 0 matching Time - min(Time) from training
            time_delta = 0.0
            #the first packet has no previous packet so time_delta is 0 matching fillna(0.0) in training
        else:
            rel_time   = ts - self.capture_start_time
            #compute time elapsed since the first packet
            time_delta = rel_time - self.prev_rel_time
            #compute time since the previous packet

        self.prev_rel_time = rel_time
        #save this packet's relative time for computing the next packet's time_delta

        feat = extract_features_from_info(info_raw, length, rel_time, time_delta)
        #extract all 12 features from the info string, length, and timing values
        self.packet_buffer.append(feat)
        #add this packet's features to the circular buffer (oldest packet is automatically dropped)
        self.packets_since_last_decision += 1
        #count this new packet

        if (len(self.packet_buffer) == SEQ_LEN and
                self.packets_since_last_decision >= STEP):
            self.packets_since_last_decision = 0
            #reset the step counter so we wait for STEP more packets before the next decision
            self._process_window()
            #fire a decision: we have a full 20-packet window and at least 5 new packets since last time

    # ── tshark line parser ────────────────────────────────────────────────────

    @staticmethod
    def parse_tshark_line(line: str) -> Optional[dict]:
        #this function parses one tab-separated line from tshark output into a structured dictionary
        try:
            parts = line.strip().split("\t")
            #split the line by tab since tshark outputs fields separated by tabs
            if len(parts) < 5:
                return None
            #if fewer than 5 fields then the line is malformed so skip it
            ts_str, src_ip, _dst_ip, len_str, info = parts[:5]
            #unpack the five expected fields: timestamp, source ip, dest ip, length, info
            if len(parts) > 5:
                info = "\t".join(parts[4:])
            #if the Info field itself contains tabs then rejoin everything from field 4 onwards
            return {
                "ts":     float(ts_str),
                #unix timestamp of the packet in seconds as a float
                "src_ip": src_ip.strip(),
                #source ip address with whitespace removed
                "info":   info.strip(),
                #the info string that contains port numbers and tcp flags
                "length": float(len_str),
                #packet byte length as a float
            }
        except Exception:
            return None
        #if any parsing step fails return None to skip this line silently

    # ── dashboard thread ──────────────────────────────────────────────────────

    def _dashboard_thread(self):
        #this function runs in a background thread and refreshes the dashboard at DISPLAY_REFRESH intervals
        while self.running:
            render_dashboard(self.stats.snapshot(), self.model_name,
                             self.det_attack_thr)
            #take a snapshot of current stats and render the dashboard
            time.sleep(DISPLAY_REFRESH)
            #wait before the next refresh

    # ── main capture loop ─────────────────────────────────────────────────────

    def run(self):
        #this function starts the dashboard thread and the main tshark capture loop
        self.running = True
        #set the running flag so the dashboard thread knows to keep going
        threading.Thread(target=self._dashboard_thread, daemon=True).start()
        #start the dashboard in a daemon thread so it exits when the main thread exits
        print("[INFO] Starting remote tshark capture...")

        remote_cmd = (
            "sudo tshark -i wlan0 -l -T fields "
            "-e frame.time_epoch "
            "-e ip.src "
            "-e ip.dst "
            "-e frame.len "
            "-e _ws.col.Info "
            "-Y \"tcp.port == 1883 or tcp.port == 8883\""
        )
        #tshark command run on the broker pi: captures tcp mqtt traffic and outputs the five fields we need
        #_ws.col.Info gives the same info string that wireshark writes to csv so our regexes work identically

        proc = subprocess.Popen(
            ["ssh",
             "-i", SSH_KEY,
             #use our ssh key for authentication
             "-o", "IdentitiesOnly=yes",
             #only use the specified key and ignore any other keys in the agent
             "-o", "StrictHostKeyChecking=no",
             #skip host key verification to avoid prompts on first connection
             "-o", "UserKnownHostsFile=/dev/null",
             #don't add the host to known hosts to keep the connection clean
             SSH_TARGET,
             #the broker pi ssh address
             remote_cmd],
            stdout=subprocess.PIPE,
            #capture tshark output so we can process it line by line
            stderr=subprocess.PIPE,
            #capture stderr so we can print ssh errors in the background thread
            text=True,
            #decode output as text instead of bytes
            bufsize=1,
            #line-buffered output so we get packets one by one without delay
        )

        print(f"[INFO] SSH started (pid {proc.pid})")
        #confirm the ssh process started and show its process id

        def _stderr(pipe):
            for ln in pipe:
                print(f"[SSH] {ln}", end="", flush=True)
        #background function that prints any ssh error messages as they arrive

        threading.Thread(target=_stderr, args=(proc.stderr,), daemon=True).start()
        #start the stderr reader in a daemon thread

        if proc.poll() is not None:
            print(f"[ERROR] SSH died immediately. rc={proc.returncode}")
            self.stop()
            return
        #safety check: if the ssh process exited immediately something went wrong so stop everything

        print("[INFO] Waiting for packets...")
        try:
            for line in proc.stdout:
                #read tshark output one line at a time (each line is one packet)
                if not self.running:
                    break
                #check the running flag so we can exit cleanly when the user presses ctrl+c
                pkt = self.parse_tshark_line(line)
                if pkt is None:
                    continue
                #skip malformed lines

                if not is_tcp_mqtt_port(pkt["info"]):
                    continue
                #apply the mqtt port filter to match the training pipeline exactly
                #tshark's bpf filter already ensures only tcp mqtt traffic but we check the info text too

                self._on_packet(
                    ts       = pkt["ts"],
                    src_ip   = pkt["src_ip"],
                    info_raw = pkt["info"],
                    length   = pkt["length"],
                )
                #process this packet: extract features, fill the buffer and trigger a decision if ready

        except KeyboardInterrupt:
            self.stop()
            #ctrl+c pressed so stop the system gracefully
        except Exception as exc:
            print(f"[ERROR] Capture loop: {exc}", file=sys.stderr)
            self.stop()
            #any other unexpected error: print it and stop

    def stop(self):
        #this function stops the capture loop, saves the adapted policy and prints a final summary
        self.running = False
        #clear the running flag to stop the capture loop and dashboard thread
        print("\n[INFO] Saving final adaptive policy...")
        self._save_adaptive_policy()
        #always save the latest policy on stop so no learning is lost
        s = self.stats.snapshot()
        print(f"[INFO] NeuroGuard Adaptive stopped.  "
              f"Packets: {s['total_packets']:,}  "
              f"Decisions: {s['total_decisions']:,}  "
              f"Online updates: {s['online_updates']:,}  "
              f"Avg reward: {s['avg_reward']:+.3f}")
        #show total packets, decisions made, gradient updates performed and average reward


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(description="NeuroGuard Adaptive — Online RL Deployment")
    pa.add_argument("--model", default="cnn_only",
                    choices=["cnn_only", "cnn_attn", "cnn_bilstm_attn"])
    #choose which detector architecture to use (default is cnn_only)
    pa.add_argument("--iface", default=DEFAULT_IFACE)
    #choose which network interface to capture on (default is wlan0)
    args = pa.parse_args()
    #parse the command line arguments

    defender = None
    #will hold the NeuroGuardAdaptive instance once it is created

    def _sig(s, f):
        print(f"\n\n{YELLOW}[INFO] Stopping and saving policy...{RESET}")
        if defender:
            defender.stop()
        sys.exit(0)
    #signal handler that stops the system cleanly and saves the policy when ctrl+c or sigterm is received

    signal.signal(signal.SIGINT,  _sig)
    #register ctrl+c handler
    signal.signal(signal.SIGTERM, _sig)
    #register termination signal handler

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"  NeuroGuard Adaptive  — Online RL Deployment")
    print(f"  Model         : {args.model}")
    print(f"  Dir           : {MODEL_DIR}")
    print(f"  Replay size   : {ONLINE_REPLAY_SIZE}")
    print(f"  Learn every   : {ONLINE_LEARN_EVERY} decisions")
    print(f"  Save every    : {ONLINE_SAVE_EVERY} decisions")
    print(f"{BOLD}{'═'*60}{RESET}\n")
    #print the startup banner with the selected model and online RL configuration

    if not MODEL_DIR.exists():
        print(f"{RED}[ERROR] Model dir not found: {MODEL_DIR}{RESET}")
        sys.exit(1)
    #safety check: if the model directory doesn't exist we can't load anything so exit immediately

    defender = NeuroGuardAdaptive(model_name=args.model, iface=args.iface)
    #create the adaptive live detection system with the chosen model and interface
    defender.run()
    #start capturing packets and making real-time adaptive decisions


if __name__ == "__main__":
    main()
    #run the main function when this script is executed directly
