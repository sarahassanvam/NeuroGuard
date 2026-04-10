# neuroguard_live_v2_online_rl.py
# VERSION 2  —  Online RL Adaptation
#
# Matches the rl_env.py reward structure exactly:
#   • attack traffic  → mitigate = +1.2, heavy mitigate penalty = -0.2,
#                        det_attack bonus = +0.3, allow = -3.0, non-mitigate = -0.8
#   • normal traffic  → ALLOW = +2.0, SAFE_ACTIONS = +0.6, block = -2.5,
#                        det_low + non-allow = -1.5, ESCALATE = -2.5
#   • action costs subtracted (same table as rl_env.py for cnn_bilstm_attn)
#
# New additions vs original v2:
#   • Full rl_env-matched reward function (replaces the 4-case stub)
#   • Drift guard: if ALLOW % drops below ALLOW_FLOOR over a sliding window,
#     the online weights are rolled back to the base checkpoint automatically
#   • fp_counter / fn_counter tracked live, fed back into the RL state so
#     the state vector exactly matches what the agent was trained on
#   • Gradient clipping (max_norm=1.0) to prevent sudden weight explosions
#   • action_costs table loaded from the same values used in rl_env.py
#
# HOW TO RUN (on Raspberry Pi):
#   python neuroguard_live_v2_online_rl.py --model cnn_bilstm_attn
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import collections
import copy
import math
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
#argparse lets us pass --model cnn_bilstm_attn from the command line when running the script
#collections gives us Counter (for counting actions) and deque (fixed-size sliding windows)
#copy gives us deepcopy to make an independent copy of the neural network for the target network
#math is for sqrt used in the attention scaling formula
#os is used to call os.system("clear") to refresh the terminal dashboard
#pickle is for loading the saved scaler (.pkl files)
#signal lets us catch Ctrl+C so we can save the model cleanly before exiting
#sys is for sys.exit() to stop the program
#threading lets us run the dashboard and RL update in background threads simultaneously
#time is for measuring how long each decision takes (response time)
#Path is for clean cross-platform file path handling
#numpy for array math, torch for neural network inference, subprocess for launching tshark

# ── PATHS AND SETTINGS ────────────────────────────────────────────────────────

MODEL_DIR   = Path("/home/client2/neuroguard_models")
ONLINE_CKPT = MODEL_DIR / "rl_online_adapted_{model}.pt"   # filled at runtime
BASE_CKPT   = MODEL_DIR / "rl_policy_best_ACCEPTED_{model}_ft.pt"  # rollback target
#MODEL_DIR is where all the model files live on the Raspberry Pi (client2 device)
#ONLINE_CKPT is where we save the continuously updated RL weights during live operation
#{model} is a placeholder that gets replaced at runtime with "cnn_only", "cnn_attn", etc.
#BASE_CKPT is the last known-good checkpoint we fall back to if the online learning goes wrong
#_ft means fine-tuned (these are the models produced by 4a_transfer_learning.py)

SEQ_LEN        = 20
STEP           = 5
STATE_DIM      = 16
N_ACTIONS      = 16
FEAT_DIM       = 12
DEFAULT_IFACE  = "wlan0"
MQTT_PORTS     = {1883, 8883}
DISPLAY_REFRESH= 0.5
#SEQ_LEN=20 means we group 20 consecutive packets into one sequence before making a decision
#STEP=5 means after each decision we wait for 5 more packets before making the next decision
#STATE_DIM=16 is the size of the state vector the RL agent receives (16 numbers describing the situation)
#N_ACTIONS=16 is the total number of possible actions the RL agent can choose from
#FEAT_DIM=12 is the number of features per packet (same 12 features from all other scripts)
#DEFAULT_IFACE="wlan0" is the WiFi interface name on the Raspberry Pi where we capture traffic
#MQTT_PORTS are the two standard MQTT ports we filter traffic on
#DISPLAY_REFRESH=0.5 means we refresh the terminal dashboard twice per second

# threat thresholds (dashboard colours only)
THRESH_LOW  = 0.20
THRESH_MID  = 0.70
THRESH_HIGH = 0.95
#these thresholds control what colour the threat bar shows on the dashboard
#det_p < 0.20 → green (normal traffic)
#det_p between 0.20 and 0.70 → yellow (suspicious, watch it)
#det_p between 0.70 and 0.95 → yellow/orange (elevated threat)
#det_p >= 0.95 → red (high threat, likely under attack)
#these are ONLY for display and do not affect what action the RL agent takes

# ── ONLINE RL SETTINGS ────────────────────────────────────────────────────────
ONLINE_BUFFER_SIZE   = 1000    # rolling experience buffer
ONLINE_UPDATE_EVERY  = 50      # gradient step every N decisions
ONLINE_BATCH         = 32
ONLINE_LR            = 1e-5    # very small — nudge, don't destroy
ONLINE_GAMMA         = 0.99
ONLINE_SAVE_EVERY    = 500     # save checkpoint every N decisions
ONLINE_TARGET_UPDATE = 200     # sync target net every N online steps
ONLINE_GRAD_CLIP     = 1.0     # max gradient norm
#ONLINE_BUFFER_SIZE=1000 means we keep the last 1000 (state, action, reward, next_state) tuples
#when the buffer is full old experiences are overwritten by new ones (ring buffer)
#ONLINE_UPDATE_EVERY=50 means we run one gradient update step after every 50 decisions
#this is slow enough to avoid making the model jump too fast on noisy live data
#ONLINE_BATCH=32 is how many experiences we randomly sample from the buffer for each update
#ONLINE_LR=1e-5 is extremely small so fine-grained nudging happens not big destructive updates
#ONLINE_GAMMA=0.99 is the discount factor that says future rewards are almost as valuable as now
#ONLINE_SAVE_EVERY=500 means we write the model weights to disk every 500 decisions
#ONLINE_TARGET_UPDATE=200 means we sync the target network every 200 gradient steps
#ONLINE_GRAD_CLIP=1.0 caps how large a single gradient update can be so no weight explosions

# reward confidence thresholds (mirrors rl_env.py thresholds)
ONLINE_CONF_HIGH = 0.50   # det_p above this  → treat as attack
ONLINE_CONF_LOW  = 0.40   # det_p below this  → treat as normal
#in live deployment we can't know the true label (we don't know if a packet is actually an attack)
#so we use the detector's confidence score (det_p) as a proxy for the true label
#if det_p >= 0.50 we assume it's an attack and reward or penalize the action accordingly
#if det_p < 0.40 we assume it's normal traffic and apply the normal-traffic reward logic
#if det_p is between 0.40 and 0.50 we are uncertain so we skip that experience (reward=0)

# drift guard — if ALLOW fraction over the last window falls below this,
# roll the online weights back to the base checkpoint
ALLOW_FLOOR        = 0.20   # at least 20% of decisions should be ALLOW
DRIFT_WINDOW       = 200    # check over the last N decisions
#drift guard protects against a failure mode called "catastrophic drift"
#this happens when online learning causes the model to become over-aggressive
#and start blocking almost everything including normal traffic
#if we look at the last 200 decisions and fewer than 20% of them were ALLOW
#it means the model is blocking too much so we roll back to the safe base checkpoint
#and clear the experience buffer so poisoned experiences don't immediately re-corrupt it

# ── ACTION IDS ───────────────────────────────────────────────────────────────

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
#these are the 16 possible actions the RL agent can take numbered 0-15
#ALLOW (0) means do nothing — let the traffic pass normally
#RATE_LIMIT (1) means slow down packets from the suspicious IP address
#TEMP_BLOCK (2) means block the IP temporarily for a short time
#PERM_BLOCK (3) means permanently block the IP (very aggressive action)
#DROP_SYN (4) means drop incoming SYN packets to defend against SYN flooding
#DROP_CONNECT (5) means reject new MQTT connection attempts
#DELAY_CONNECT (6) means make clients wait before connecting (slows flood attacks)
#LIMIT_PUBLISH (7) means restrict how many messages a client can publish per second
#BLOCK_SUBSCRIBE (8) means stop the client from subscribing to new topics
#DISCONNECT (9) means forcefully disconnect the client from the broker
#QUARANTINE (10) means move the client to an isolated zone for monitoring
#ISOLATE_NODE (11) means cut off an entire node from the network
#REDUCE_QOS (12) means lower the quality of service level to reduce broker load
#ALERT_ONLY (13) means log a warning but take no blocking action
#ESCALATE (14) means raise the alert level (escalation_level increases by 1)
#DEESCALATE (15) means lower the alert level when the threat appears to have passed

ACTION_NAMES = {
    0: "ALLOW", 1: "RATE_LIMIT_IP", 2: "TEMP_BLOCK_IP", 3: "PERM_BLOCK_IP",
    4: "DROP_SYN_DELAY_TCP", 5: "DROP_CONNECT", 6: "DELAY_CONNECT",
    7: "LIMIT_PUBLISH", 8: "BLOCK_SUBSCRIBE", 9: "DISCONNECT_CLIENT",
    10: "QUARANTINE_CLIENT", 11: "ISOLATE_NODE", 12: "REDUCE_QOS",
    13: "ALERT_ONLY", 14: "ESCALATE", 15: "DEESCALATE",
}
#human-readable names for each action ID used in the dashboard and log prints

SAFE_ACTIONS         = {A_ALLOW, A_ALERT_ONLY, A_DEESCALATE}
HEAVY_ACTIONS        = {A_TEMP_BLOCK, A_PERM_BLOCK, A_DISCONNECT, A_QUARANTINE, A_ISOLATE_NODE}
NON_MITIGATE_ACTIONS = {A_ALLOW, A_ALERT_ONLY, A_ESCALATE, A_DEESCALATE}
LIGHT_MITIGATIONS    = {
    A_RATE_LIMIT, A_DROP_SYN, A_DELAY_CONNECT,
    A_DROP_CONNECT, A_LIMIT_PUBLISH, A_BLOCK_SUBSCRIBE, A_REDUCE_QOS,
}
#SAFE_ACTIONS are appropriate when traffic is normal — they don't block anything important
#HEAVY_ACTIONS are aggressive and disruptive so they get a small extra penalty even when correct
#NON_MITIGATE_ACTIONS don't actually stop an attack so they are penalized if an attack is happening
#LIGHT_MITIGATIONS are moderate responses — they reduce load without fully blocking a client

# action costs — identical to rl_env.py standard (cnn_bilstm_attn / cnn_attn) table
ACTION_COSTS = {
    A_ALLOW:          0.00,
    A_RATE_LIMIT:     0.15,
    A_DROP_SYN:       0.20,
    A_DROP_CONNECT:   0.30,
    A_DELAY_CONNECT:  0.20,
    A_LIMIT_PUBLISH:  0.25,
    A_BLOCK_SUBSCRIBE:0.25,
    A_REDUCE_QOS:     0.20,
    A_ALERT_ONLY:     0.05,
    A_ESCALATE:       0.40,
    A_DEESCALATE:     0.10,
    A_TEMP_BLOCK:     0.60,
    A_PERM_BLOCK:     1.00,
    A_DISCONNECT:     0.80,
    A_QUARANTINE:     1.00,
    A_ISOLATE_NODE:   1.20,
}
#action costs represent the "price" of taking an action in terms of disruption to normal operations
#ALLOW costs 0.00 because it causes no disruption at all
#ALERT_ONLY costs 0.05 because logging has a tiny overhead
#light mitigations (0.15-0.30) cause moderate disruption so they cost a little
#heavy actions like PERM_BLOCK and QUARANTINE cost 0.60-1.20 because they are very disruptive
#ISOLATE_NODE is the most expensive (1.20) because it cuts off an entire network segment
#these costs are subtracted from the reward every step so the agent is incentivized
#to use lighter actions when possible and only escalate to heavy actions when clearly necessary
#these values MUST match rl_env.py exactly so online learning stays consistent with training

# ── COLOURS ───────────────────────────────────────────────────────────────────

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
#these are ANSI terminal escape codes that change the color of printed text
#the \033[ starts the escape sequence and the number is the color code
#RESET must be added at the end of any colored text to return to normal color
#without RESET everything printed after would stay the same color

def colour_action(action_id: int, name: str) -> str:
    if action_id == A_ALLOW:
        return f"{GREEN}{name}{RESET}"
    elif action_id in HEAVY_ACTIONS:
        return f"{RED}{BOLD}{name}{RESET}"
    elif action_id in NON_MITIGATE_ACTIONS:
        return f"{YELLOW}{name}{RESET}"
    else:
        return f"{CYAN}{name}{RESET}"
#this function wraps an action name in the right color codes for the dashboard
#ALLOW → green because it means traffic is safe
#HEAVY_ACTIONS (temp block, perm block etc.) → red+bold because they are serious responses
#NON_MITIGATE_ACTIONS (escalate, alert only) → yellow as a caution indicator
#everything else (light mitigations) → cyan

def bar(value: float, width: int = 20) -> str:
    filled  = max(0, min(width, int(round(value * width))))
    bar_str = "█" * filled + "░" * (width - filled)
    if value >= 0.70:
        return f"{RED}{bar_str}{RESET}"
    elif value >= 0.30:
        return f"{YELLOW}{bar_str}{RESET}"
    else:
        return f"{GREEN}{bar_str}{RESET}"
#this draws a visual progress bar using block characters (█ for filled, ░ for empty)
#the bar width is 20 characters and value goes from 0.0 to 1.0
#it is colored based on the threat level: green for low, yellow for medium, red for high
#this makes the dashboard easy to read at a glance without looking at numbers


# ── MODEL DEFINITIONS ─────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        #must be divisible so each head gets an equal share of the dimensions
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        #each head independently attends to a slice of the feature space
        #4 heads × 32 dims each = 128 total dimensions
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        #Query (what am I looking for?), Key (what do I have?), Value (what do I share?)
        #these three projections are the core of the attention mechanism
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        #combines all heads back into one output vector
        self.scale     = math.sqrt(self.head_dim)
        #dividing by sqrt(head_dim) prevents dot products from becoming too large
        #which would push softmax into a flat distribution and slow learning

    def forward(self, x):
        B, T, D = x.shape
        #B=batch size, T=time steps (20 packets), D=feature dimension (128)
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        #project then reshape so each head works on its own slice independently
        #transpose(1,2) puts heads before time so matrix operations work correctly
        scores       = (Q @ K.transpose(-2, -1)) / self.scale
        #dot product between Q and K gives attention scores
        #high score means "packet i should pay attention to packet j"
        attn_weights = torch.softmax(scores, dim=-1)
        #convert scores to probabilities summing to 1 across the time dimension
        out          = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, D)
        #weighted sum of Values using the attention probabilities
        #then reshape back to (B, T, D)
        return self.out_proj(out).mean(dim=1)
        #final projection then average over time to get one vector per sequence


class CNN_Only(nn.Module):
    def __init__(self, feat_dim=12, seq_len=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        #two 1D convolutional layers slide a filter of size 3 over the 20-packet window
        #this detects local patterns like "SYN followed by ACK" in adjacent packets
        #128 filters per layer means 128 different local patterns are learned
        #BatchNorm stabilizes training by normalizing outputs after each conv layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * seq_len, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        #Flatten collapses (128, 20) into 2560 then Linear compresses to 128
        #Dropout(0.3) turns off 30% of neurons randomly to prevent overfitting
        #final Linear(128,1) gives a single attack probability score
    def forward(self, x):
        return self.fc(self.conv(x.transpose(1, 2))).squeeze(1)
        #transpose(1,2) because Conv1d wants (batch, channels, length) but input is (batch, length, channels)


class CNN_Attention(nn.Module):
    def __init__(self, feat_dim=12, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        #same local feature extractor as CNN_Only
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        #attention layer lets the model look at the entire 20-packet window globally
        #and decide which packets matter most for the current decision
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        #run through conv then transpose back for attention which wants (B, T, D)
        return self.fc(self.multi_head_attn(h)).squeeze(1)


class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self, feat_dim=12, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        #BiLSTM reads the sequence forward (left to right) AND backward (right to left)
        #forward hidden size is 64 and backward is also 64 so total output is 128 per step
        #this captures long-range dependencies like "there were many SYNs 10 packets ago"
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        #attention on top of BiLSTM so the model can also weight which time steps matter most
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        h, _ = self.bilstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        #_ is the final hidden state which we don't need since attention handles summarization
        return self.fc(self.multi_head_attn(h)).squeeze(1)


class QNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    def forward(self, x):
        return self.net(x)
#QNet is the RL agent's brain — it is a simple 3-layer fully connected network
#input is 16 numbers (the state vector describing current network conditions)
#output is 16 numbers (one Q-value for each possible action)
#the Q-value represents "how much total future reward do I expect if I take this action?"
#the agent always picks the action with the highest Q-value (greedy policy)
#hidden layers of size 256 give enough capacity to learn complex action-value relationships


# ── REWARD FUNCTION — matches rl_env.py exactly ───────────────────────────────

def compute_live_reward(
    det_p: float,
    action: int,
    fp_counter: int,
    fn_counter: int,
    escalation_level: int,
) -> float:
    """
    Mirrors rl_env.py NeuroGuardRLEnv.step() reward logic for cnn_bilstm_attn.

    We cannot observe y_true in the live setting, so we use detector confidence
    as a proxy:
        det_p >= ONLINE_CONF_HIGH  →  treat as attack  (y_true == 1 branch)
        det_p <  ONLINE_CONF_LOW   →  treat as normal   (y_true == 0 branch)
        in between                 →  ambiguous, return 0.0 (skip)

    Action costs and counter dynamics are identical to rl_env.py.
    """
    is_allow     = (action == A_ALLOW)
    is_mitigate  = (action not in NON_MITIGATE_ACTIONS)
    det_attack   = (det_p >= ONLINE_CONF_HIGH)
    #is_allow: True if the agent chose to do nothing (pass the traffic)
    #is_mitigate: True if the agent chose any actual blocking or slowing action
    #det_attack: True if the detector is confident this is an attack (det_p >= 0.50)

    reward = 0.0

    # ── attack branch ─────────────────────────────────────────────────────────
    if det_p >= ONLINE_CONF_HIGH:
        #we treat this sequence as an attack because det_p >= 0.50
        if is_allow:
            reward -= 3.0
            #the worst possible action on an attack is to ALLOW it through
            #-3.0 is a large penalty to strongly discourage this mistake
        elif action in NON_MITIGATE_ACTIONS:   # ESCALATE / ALERT_ONLY / DEESCALATE
            reward -= 0.8
            #escalating or just alerting is better than allowing but still doesn't stop the attack
            #-0.8 is a moderate penalty to push the agent toward actually blocking
        else:
            reward += 1.2
            #+1.2 is the main reward for correctly mitigating an attack
            if det_attack:
                reward += 0.3
                #+0.3 bonus when the detector is also confident it's an attack (double confirmation)
            if action in HEAVY_ACTIONS:
                reward -= 0.2
                #-0.2 small penalty for using very heavy actions (like permanent block)
                #this encourages the agent to prefer lighter mitigations unless truly necessary

    # ── normal branch ─────────────────────────────────────────────────────────
    elif det_p < ONLINE_CONF_LOW:
        #we treat this sequence as normal traffic because det_p < 0.40
        if is_allow:
            reward += 2.0
            #+2.0 is the main reward for correctly allowing normal traffic through
        elif action in SAFE_ACTIONS:           # ALERT_ONLY or DEESCALATE
            reward += 0.6
            #+0.6 if we chose a safe non-blocking action on normal traffic (not perfect but acceptable)
        else:
            reward -= 2.5
            #-2.5 large penalty for blocking normal traffic (false positive)
            #this strongly discourages the agent from being over-aggressive

        # extra penalty when detector is very confident of normal
        if det_p < ONLINE_CONF_LOW and not is_allow:
            reward -= 1.5
            #-1.5 additional penalty when we are confident it's normal but still block it
            #double punishment for being clearly wrong

        # punish ESCALATE on normal — same as rl_env.py
        if action == A_ESCALATE:
            reward -= 2.5
            #raising the alert level when traffic is clearly normal is strongly penalized
            #because it leads to a spiral of unnecessary blocking

    # ── ambiguous zone ────────────────────────────────────────────────────────
    else:
        return 0.0   # don't learn from uncertain observations
    #if det_p is between 0.40 and 0.50 we don't know if it's attack or normal
    #so we skip this experience and don't add it to the buffer to avoid teaching the agent
    #from uncertain observations that could push it in the wrong direction

    # subtract action cost (same table as rl_env.py)
    reward -= ACTION_COSTS.get(action, 0.0)
    #subtract the disruption cost of the chosen action
    #heavier actions cost more so the agent is incentivized to use the lightest effective action

    return float(reward)


# ── ONLINE RL UPDATER ─────────────────────────────────────────────────────────

class OnlineRLUpdater:
    """
    Background thread: collects live (s,a,r,ns) tuples and runs one DQN
    gradient step every ONLINE_UPDATE_EVERY decisions.

    Additions vs original:
      • Gradient clipping (ONLINE_GRAD_CLIP) — prevents weight explosions.
      • Drift guard: if ALLOW fraction in the recent-action window falls below
        ALLOW_FLOOR, the Q-net weights are rolled back to the base checkpoint.
      • skip reward==0 tuples (ambiguous zone) — keeps the buffer clean.
    """

    def __init__(self, q_net: QNet, device: str,
                 online_ckpt_path: Path, base_ckpt_path: Path):
        self.device            = device
        self.online_ckpt_path  = online_ckpt_path
        #where to save the updated online weights periodically
        self.base_ckpt_path    = base_ckpt_path
        #the safe fallback checkpoint for the drift rollback mechanism
        self._lock             = threading.Lock()
        #a lock to prevent two threads from updating the Q-network weights at the same time
        #without this the live thread and update thread could corrupt the weights

        self.q_net  = q_net
        self.tq_net = copy.deepcopy(q_net)
        self.tq_net.eval()
        #tq_net is the TARGET NETWORK — a frozen copy of q_net
        #in DQN we use a target network to compute the "expected future reward" (next_q)
        #if we used q_net for both current and future Q-values the targets would keep changing
        #which makes training unstable (like chasing a moving target)
        #we sync tq_net with q_net every ONLINE_TARGET_UPDATE steps

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=ONLINE_LR)
        self.loss_fn   = nn.SmoothL1Loss()
        #SmoothL1Loss (also called Huber loss) is used instead of MSE because
        #it is less sensitive to large errors so training is more stable on noisy live data

        # ring buffer — stores the last ONLINE_BUFFER_SIZE experiences
        self.buf_s  = np.zeros((ONLINE_BUFFER_SIZE, STATE_DIM), dtype=np.float32)
        self.buf_a  = np.zeros((ONLINE_BUFFER_SIZE,),           dtype=np.int64)
        self.buf_r  = np.zeros((ONLINE_BUFFER_SIZE,),           dtype=np.float32)
        self.buf_ns = np.zeros((ONLINE_BUFFER_SIZE, STATE_DIM), dtype=np.float32)
        self.buf_d  = np.zeros((ONLINE_BUFFER_SIZE,),           dtype=np.float32)
        self.buf_i  = 0
        self.buf_n  = 0
        #buf_s stores states, buf_a stores actions, buf_r stores rewards
        #buf_ns stores next states, buf_d stores done flags (1 if episode ended)
        #buf_i is the current write position in the ring (wraps around when full)
        #buf_n is how many valid entries are currently in the buffer (up to ONLINE_BUFFER_SIZE)

        # counters / diagnostics
        self.total_updates  = 0
        self.online_step    = 0
        self.last_loss      = 0.0
        self.rollback_count = 0
        #total_updates counts how many gradient steps have been taken in total
        #online_step counts steps since the last target network sync
        #last_loss is the most recent training loss value shown in the dashboard
        #rollback_count counts how many times the drift guard triggered a rollback

        # pending queue (written by live thread, flushed by update thread)
        self._pending      = []
        self._pending_lock = threading.Lock()
        #the live thread (packet processing) and update thread run simultaneously
        #_pending is a temporary list where the live thread drops new experiences
        #the update thread grabs them using _flush_pending() and moves them to the ring buffer
        #_pending_lock ensures these two threads don't both write/read _pending at the same time

        # sliding window of recent actions for drift detection
        self._recent_actions = collections.deque(maxlen=DRIFT_WINDOW)
        #deque with maxlen=200 automatically drops old actions when full
        #we use this to calculate the ALLOW fraction over the last 200 decisions

    # ── called by live thread after every decision ────────────────────────────

    def push(self, state, action, reward, next_state, done=False):
        if reward == 0.0:
            return   # skip ambiguous — don't pollute the buffer
        #if reward is 0 it means the detector was uncertain (ambiguous zone)
        #we don't add these to the buffer because they would confuse the learning signal
        with self._pending_lock:
            self._pending.append((
                state.copy(), int(action), float(reward),
                next_state.copy(), float(done),
            ))
        #add the experience to the pending list so the update thread can pick it up
        #state.copy() is important because the state array will be modified later
        #without copy() both the buffer entry and the live state would point to the same array
        self._recent_actions.append(action)
        #record this action in the sliding window for drift detection

    # ── internal helpers ──────────────────────────────────────────────────────

    def _flush_pending(self):
        with self._pending_lock:
            pending, self._pending = self._pending, []
            #atomically grab the pending list and reset it to empty
            #this is thread-safe because we hold the lock during the swap
        for s, a, r, ns, d in pending:
            self.buf_s[self.buf_i]  = s
            self.buf_a[self.buf_i]  = a
            self.buf_r[self.buf_i]  = r
            self.buf_ns[self.buf_i] = ns
            self.buf_d[self.buf_i]  = d
            self.buf_i = (self.buf_i + 1) % ONLINE_BUFFER_SIZE
            #wrap around using modulo so old entries get overwritten when buffer is full
            self.buf_n = min(self.buf_n + 1, ONLINE_BUFFER_SIZE)
            #track how many valid entries are in the buffer (caps at ONLINE_BUFFER_SIZE)

    def _check_drift_and_rollback(self):
        """
        If the model has started blocking almost everything (ALLOW fraction too
        low), roll the online weights back to the base accepted checkpoint.
        This is the drift guard described in the design note.
        """
        if len(self._recent_actions) < DRIFT_WINDOW:
            return   # not enough data yet
        #we need a full window of 200 decisions before we can make a reliable judgment
        allow_frac = self._recent_actions.count(A_ALLOW) / len(self._recent_actions)
        #calculate what fraction of the last 200 decisions were ALLOW
        if allow_frac < ALLOW_FLOOR:
            #if fewer than 20% of decisions are ALLOW the model is being too aggressive
            print(
                f"\n[ONLINE-RL] DRIFT GUARD triggered — ALLOW fraction={allow_frac:.2f} "
                f"< floor={ALLOW_FLOOR}. Rolling back to base checkpoint.",
                flush=True,
            )
            with self._lock:
                base_state = torch.load(self.base_ckpt_path, map_location=self.device)
                self.q_net.load_state_dict(base_state)
                self.tq_net.load_state_dict(base_state)
                self.q_net.eval()
            #load the last known-good (base) weights into both Q-net and target Q-net
            #we hold the lock to make sure the live thread doesn't read the weights
            #in the middle of this rollback operation
            # clear the buffer so stale "block-everything" experiences don't
            # immediately re-poison the next update
            self.buf_n = 0
            self.buf_i = 0
            self._recent_actions.clear()
            self.rollback_count += 1
            #reset the buffer position and count so the ring buffer starts fresh
            #clear the recent actions deque so we start a fresh drift window
            print(f"[ONLINE-RL] Rollback #{self.rollback_count} complete.", flush=True)

    def _do_update(self):
        if self.buf_n < ONLINE_BATCH:
            return
        #we need at least 32 experiences in the buffer before we can sample a batch
        #if we have fewer we skip the update and wait for more data to arrive

        idx = np.random.randint(0, self.buf_n, size=ONLINE_BATCH)
        #randomly sample 32 indices from the buffer (experience replay)
        #random sampling breaks the correlation between consecutive experiences
        #which makes training more stable and prevents the model from overfitting
        #to recent patterns
        bs  = torch.tensor(self.buf_s[idx]).to(self.device)
        ba  = torch.tensor(self.buf_a[idx]).to(self.device)
        br  = torch.tensor(self.buf_r[idx]).to(self.device)
        bns = torch.tensor(self.buf_ns[idx]).to(self.device)
        bd  = torch.tensor(self.buf_d[idx]).to(self.device)
        #convert numpy arrays to PyTorch tensors and move to device (CPU on Raspberry Pi)

        with self._lock:
            self.q_net.train()
            #switch to training mode so gradients are computed

            qsa = self.q_net(bs).gather(1, ba.view(-1, 1)).squeeze(1)
            #compute Q-values for the batch of states then select only the Q-value
            #for the action that was actually taken (using gather)
            #this is the current estimate of "how good was that action?"

            with torch.no_grad():
                best_a = torch.argmax(self.q_net(bns), dim=1)
                next_q = self.tq_net(bns).gather(1, best_a.view(-1, 1)).squeeze(1)
                target = br + ONLINE_GAMMA * (1.0 - bd) * next_q
            #Double DQN: use q_net to SELECT the best next action
            #but use tq_net (frozen) to EVALUATE that action's Q-value
            #this reduces overestimation bias compared to plain DQN
            #target is the Bellman equation: r + gamma * Q(next_state, best_action)
            #(1 - bd) zeroes out the future reward for terminal states

            loss = self.loss_fn(qsa, target)
            #Huber loss between current Q estimate and Bellman target
            #if loss goes down the model is correctly learning which actions lead to more reward
            self.optimizer.zero_grad()
            loss.backward()
            #backpropagate to compute gradients for all Q-net parameters
            # gradient clipping — prevents large weight jumps on Pi
            nn.utils.clip_grad_norm_(self.q_net.parameters(), ONLINE_GRAD_CLIP)
            #clip gradients so no single update can change weights by more than 1.0
            #this is especially important on the Raspberry Pi where we want stable operation
            self.optimizer.step()
            self.q_net.eval()
            #apply the gradient update then switch back to eval mode for inference

        self.last_loss     = float(loss.item())
        self.total_updates += 1
        self.online_step   += 1

        # sync target net
        if self.online_step % ONLINE_TARGET_UPDATE == 0:
            self.tq_net.load_state_dict(self.q_net.state_dict())
            #every 200 gradient steps copy the current Q-net weights into the target network
            #this gives the target a stable reference while the Q-net keeps updating

        # periodic checkpoint save
        if self.online_step % ONLINE_SAVE_EVERY == 0:
            with self._lock:
                torch.save(self.q_net.state_dict(), self.online_ckpt_path)
            print(
                f"\n[ONLINE-RL] Checkpoint saved → {self.online_ckpt_path}  "
                f"(updates={self.total_updates}  rollbacks={self.rollback_count})",
                flush=True,
            )
            #save the weights to disk every 500 gradient steps so if the Pi loses power
            #we don't lose all the online learning progress

    # ── background update loop ────────────────────────────────────────────────

    def run_update_loop(self, decision_counter_ref, running_ref):
        last_triggered_at = 0
        while running_ref[0]:
            current = decision_counter_ref[0]
            if current - last_triggered_at >= ONLINE_UPDATE_EVERY:
                #check if we have made ONLINE_UPDATE_EVERY=50 new decisions since last update
                self._flush_pending()
                #move new experiences from the pending list into the ring buffer
                self._check_drift_and_rollback()
                #check if the model has drifted and needs to be rolled back
                self._do_update()
                #run one gradient step if the buffer has enough data
                last_triggered_at = current
            time.sleep(0.1)
            #sleep briefly between checks to avoid burning 100% CPU on the Pi

    @property
    def buffer_fill(self):
        return self.buf_n
    #property so the dashboard can read buf_n without accessing the internal variable directly


# ── LIVE STATE TRACKER — matches rl_env.py _make_state exactly ───────────────

class LiveStateTracker:
    """
    Builds the 16-dim state vector in exactly the same way as
    rl_env.py NeuroGuardRLEnv._make_state(), so the live state distribution
    matches what the agent was trained on.

    Also maintains fp_counter and fn_counter that update after each decision
    based on detector confidence (proxy for true label).
    """
    def __init__(self):
        self.prev_action      = A_ALLOW
        self.escalation_level = 0
        self.fp_counter       = 0
        self.fn_counter       = 0
        #prev_action starts as ALLOW (0) which is a neutral starting state
        #escalation_level starts at 0 (no alert raised yet)
        #fp_counter tracks how many probable false positives have happened recently
        #fn_counter tracks how many probable missed attacks have happened recently
        #these counters are part of the 16-dim state vector so the agent can
        #adjust its behavior based on its own recent mistake history

    def build_state(self, seq_x_scaled: np.ndarray, det_p: float) -> np.ndarray:
        # indices match rl_env.py _make_state exactly
        time_delta_mean = float(np.mean(seq_x_scaled[:, 1]))
        length_mean     = float(np.mean(seq_x_scaled[:, 2]))
        has_mqtt_mean   = float(np.mean(seq_x_scaled[:, 3]))
        # flags: indices 4–11  (syn ack fin rst psh urg to_mqtt from_mqtt)
        flags_mean = np.mean(seq_x_scaled[:, 4:12], axis=0).astype(np.float32)
        #we compute the mean of each feature across the 20-packet window
        #so the state represents the average characteristics of the recent traffic burst
        #this is exactly what rl_env.py does so the live state matches training state distribution

        extras = np.array([
            det_p,
            time_delta_mean,
            length_mean,
            has_mqtt_mean,
            float(self.prev_action)      / 15.0,
            float(self.escalation_level) / 5.0,
            float(self.fp_counter)       / 10.0,
            float(self.fn_counter)       / 10.0,
        ], dtype=np.float32)
        #det_p is the detector's attack probability (already between 0 and 1)
        #prev_action is normalized by dividing by 15 (max action ID) to put it in [0,1]
        #escalation_level is normalized by dividing by 5 (max level) to put it in [0,1]
        #fp_counter and fn_counter are normalized by dividing by 10 (their max) to put in [0,1]
        #normalizing all state values to a similar range helps the neural network learn faster

        return np.concatenate([extras, flags_mean]).astype(np.float32)
        #8 extras + 8 flag means = 16 total values → the 16-dim state vector

    def update(self, action: int, det_p: float):
        """Update internal counters — mirrors rl_env.py step() logic."""
        # escalation tracking
        if action == A_ESCALATE:
            self.escalation_level = min(self.escalation_level + 1, 5)
        elif action == A_DEESCALATE:
            self.escalation_level = max(self.escalation_level - 1, 0)
        #escalation_level goes up when ESCALATE is chosen (max 5 to prevent overflow)
        #it comes down when DEESCALATE is chosen (min 0 to prevent negative)
        #this counter is fed back into the next state so the agent knows its current alert level

        # fp / fn counter updates (proxy using detector confidence)
        if det_p >= ONLINE_CONF_HIGH:
            # likely attack
            if action in NON_MITIGATE_ACTIONS:
                self.fn_counter = min(self.fn_counter + 1, 10)
            else:
                self.fn_counter = max(self.fn_counter - 1, 0)
        #if traffic looks like an attack (det_p >= 0.50) and we didn't mitigate
        #we count it as a probable missed attack (false negative) so fn_counter goes up
        #if we did mitigate fn_counter goes down because we're handling attacks correctly
        elif det_p < ONLINE_CONF_LOW:
            # likely normal
            if action == A_ALLOW:
                self.fp_counter = max(self.fp_counter - 2, 0)
            elif action in SAFE_ACTIONS:
                self.fp_counter = max(self.fp_counter - 1, 0)
            else:
                self.fp_counter = min(self.fp_counter + 2, 10)
        #if traffic looks normal (det_p < 0.40) and we blocked it → probable false positive
        #fp_counter goes up by 2 (larger jump to make the agent more cautious faster)
        #if we allowed or took a safe action fp_counter goes down

        self.prev_action = action
        #save the current action as prev_action so the next state includes it


# ── LIVE STATS ────────────────────────────────────────────────────────────────

class LiveStats:
    def __init__(self):
        self._lock             = threading.Lock()
        #the lock prevents the dashboard thread and live thread from reading/writing
        #the same counters at the same time which would cause corrupted values
        self.total_packets     = 0
        self.total_decisions   = 0
        self.attack_decisions  = 0
        self.normal_decisions  = 0
        self.action_counts     = collections.Counter()
        self.recent_detp       = collections.deque(maxlen=50)
        #stores the last 50 detector probability values for computing avg threat level on dashboard
        self.recent_actions    = collections.deque(maxlen=10)
        #stores the last 10 decisions shown in the dashboard "Last decisions" section
        self.recent_src_ips    = collections.deque(maxlen=5)
        #stores the last 5 source IP addresses seen in the traffic
        self.start_time        = time.time()
        self.response_times_ms = collections.deque(maxlen=200)
        #stores the last 200 response times for computing avg/min/max/p95 on dashboard
        # online RL diagnostics
        self.online_updates    = 0
        self.online_buf_fill   = 0
        self.last_online_loss  = 0.0
        self.online_rollbacks  = 0

    def record_packet(self, src_ip: str):
        with self._lock:
            self.total_packets += 1
            self.recent_src_ips.append(src_ip)
        #called every time a new packet arrives to update the packet counter and IP list

    def record_decision(self, det_p, action_id, action_name, response_ms):
        with self._lock:
            self.total_decisions += 1
            self.recent_detp.append(det_p)
            self.action_counts[action_name] += 1
            self.recent_actions.append((action_id, action_name, det_p))
            self.response_times_ms.append(response_ms)
            if det_p >= THRESH_HIGH:
                self.attack_decisions += 1
            elif det_p < THRESH_LOW:
                self.normal_decisions += 1
        #called after every sequence is processed and a decision is made
        #we use the lock here so the dashboard thread always sees consistent values

    def update_online_stats(self, updates, buf_fill, last_loss, rollbacks):
        with self._lock:
            self.online_updates   = updates
            self.online_buf_fill  = buf_fill
            self.last_online_loss = last_loss
            self.online_rollbacks = rollbacks
        #called after each online RL update to push fresh diagnostics to the dashboard

    def snapshot(self):
        with self._lock:
            rt = list(self.response_times_ms)
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
                avg_response_ms  = float(np.mean(rt))            if rt else 0.0,
                min_response_ms  = float(np.min(rt))             if rt else 0.0,
                max_response_ms  = float(np.max(rt))             if rt else 0.0,
                p95_response_ms  = float(np.percentile(rt, 95))  if rt else 0.0,
                online_updates   = self.online_updates,
                online_buf_fill  = self.online_buf_fill,
                last_online_loss = self.last_online_loss,
                online_rollbacks = self.online_rollbacks,
            )
        #snapshot() takes a consistent copy of all stats under the lock
        #and returns a plain dictionary so the dashboard thread can read
        #the values without holding the lock for a long time


# ── DASHBOARD ─────────────────────────────────────────────────────────────────

def render_dashboard(stats: dict, model_name: str, iface: str):
    os.system("clear")
    #clear the terminal before redrawing so the dashboard refreshes in place
    uptime    = stats["uptime_s"]
    hrs, rem  = divmod(int(uptime), 3600)
    mins, sec = divmod(rem, 60)
    #calculate hours, minutes and seconds from total uptime in seconds
    pkt_rate  = stats["total_packets"]   / max(uptime, 1)
    dec_rate  = stats["total_decisions"] / max(uptime, 1)
    #packets per second and decisions per second since startup
    detp_list = stats["recent_detp"]
    avg_detp  = float(np.mean(detp_list)) if detp_list else 0.0
    #average detector probability over the last 50 decisions to get a smooth threat level

    if avg_detp >= THRESH_HIGH:
        threat_label = f"{RED}{BOLD}! HIGH THREAT  !{RESET}"
    elif avg_detp >= THRESH_MID:
        threat_label = f"{YELLOW} ELEVATED{RESET}"
    elif avg_detp >= THRESH_LOW:
        threat_label = f"{YELLOW}~ SUSPICIOUS{RESET}"
    else:
        threat_label = f"{GREEN}  NORMAL{RESET}"
    #choose the right threat label and color based on the average detector score

    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"  {CYAN}{BOLD}NeuroGuard Live  v2  |  Model: {model_name}{RESET}")
    print(f"  Uptime: {hrs:02d}:{mins:02d}:{sec:02d}  │  Ctrl+C to stop")
    print(f"{'═'*60}")
    print(f"  Packets captured : {WHITE}{stats['total_packets']:,}{RESET}  ({pkt_rate:.1f} pkt/s)")
    print(f"  Decisions made   : {WHITE}{stats['total_decisions']:,}{RESET}  ({dec_rate:.1f} dec/s)")
    print()
    print(f"  Threat level  {bar(avg_detp)}  {avg_detp:.2f}  {threat_label}")
    print()

    # online RL status
    buf_pct  = 100.0 * stats["online_buf_fill"] / ONLINE_BUFFER_SIZE
    loss_str = f"{stats['last_online_loss']:.4f}" if stats["online_updates"] > 0 else "—"
    rb_str   = (f"  {YELLOW}Rollbacks: {stats['online_rollbacks']}{RESET}"
                if stats["online_rollbacks"] > 0 else "")
    print(f"  {BOLD}Online RL Adaptation:{RESET}")
    print(f"  Updates: {WHITE}{stats['online_updates']}{RESET}  "
          f"Buffer: {WHITE}{stats['online_buf_fill']}/{ONLINE_BUFFER_SIZE}{RESET} ({buf_pct:.0f}%)  "
          f"Loss: {WHITE}{loss_str}{RESET}{rb_str}")
    print(f"  Update interval: every {ONLINE_UPDATE_EVERY} decisions  "
          f"(batch={ONLINE_BATCH}  LR={ONLINE_LR}  clip={ONLINE_GRAD_CLIP})")
    print()
    #show how many gradient updates have been done, how full the buffer is, and the loss
    #rollbacks are shown in yellow as a warning if any happened

    print(f"  {BOLD}Response time (per {SEQ_LEN}-packet window → 1 decision):{RESET}")
    print(f"  Avg : {WHITE}{stats['avg_response_ms']:.3f} ms{RESET}  "
          f"Min : {GREEN}{stats['min_response_ms']:.3f} ms{RESET}  "
          f"Max : {YELLOW}{stats['max_response_ms']:.3f} ms{RESET}")
    print(f"  p95 : {WHITE}{stats['p95_response_ms']:.3f} ms{RESET}")
    if stats["avg_response_ms"] > 0:
        print(f"  Throughput: {WHITE}{1000.0/stats['avg_response_ms']:.1f} dec/s{RESET}")
    print()
    #p95 latency means 95% of all decisions were faster than this value
    #this is the real-time performance guarantee — if p95 is under a few milliseconds
    #the system is fast enough for real-time network protection

    print(f"  {BOLD}Last decisions:{RESET}")
    for (aid, aname, dp) in reversed(list(stats["recent_actions"])):
        print(f"    det_p={dp:.2f}  →  {colour_action(aid, aname)}")
    print()
    #show the 10 most recent decisions in reverse order (newest first) with their confidence

    print(f"  {BOLD}Top actions (all time):{RESET}")
    total_dec = max(stats["total_decisions"], 1)
    for aname, cnt in sorted(stats["action_counts"].items(), key=lambda x: -x[1])[:5]:
        pct = 100.0 * cnt / total_dec
        aid = next((k for k, v in ACTION_NAMES.items() if v == aname), 0)
        print(f"    {colour_action(aid, f'{aname:25s}')}  {cnt:6d}  ({pct:5.1f}%)")
    print()
    #show the 5 most frequently chosen actions with their count and percentage
    #sorted in descending order by count (most used first)

    if stats["recent_src_ips"]:
        print(f"  {BOLD}Recent source IPs:{RESET}")
        for ip in set(list(stats["recent_src_ips"])[-5:]):
            print(f"    {WHITE}{ip}{RESET}")
        print()
    #show the last few unique source IP addresses seen in the traffic
    #set() removes duplicates so we don't show the same IP multiple times

    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"  {RED}High-threat{RESET}={stats['attack_decisions']}  "
          f"{GREEN}Normal{RESET}={stats['normal_decisions']}  "
          f"Uncertain={stats['total_decisions']-stats['attack_decisions']-stats['normal_decisions']}")
    print(f"{'═'*60}")
    #final summary line showing how many decisions were classified as high threat, normal, or uncertain


# ── MAIN DEFENDER CLASS ───────────────────────────────────────────────────────

class NeuroGuardLive:

    def __init__(self, model_name: str, iface: str):
        self.model_name = model_name
        self.iface      = iface
        self.device     = "cpu"
        #we always use CPU on the Raspberry Pi since it has no GPU
        self.running    = [True]
        #using a list instead of a plain bool so background threads can read and modify
        #it through their reference (Python doesn't let you pass a bool by reference)

        # load scaler — fine-tuned
        name_map = {
            "cnn_only":        "scaler_cnn_only_ft.pkl",
            "cnn_attn":        "scaler_cnn_attention_ft.pkl",
            "cnn_bilstm_attn": "scaler_cnn_bilstm_attn_ft.pkl",
        }
        scaler_path = MODEL_DIR / name_map[model_name]
        print(f"[INFO] Loading scaler from {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        #load the scaler that was re-fit on real hardware data in 4a_transfer_learning.py
        #this normalizer knows the real hardware's time delta and packet size distributions
        #using the wrong scaler would give the model wrongly scaled inputs and ruin detection

        # load detector — fine-tuned, always frozen
        ckpt_map = {
            "cnn_only":        "detector_cnn_only_ft.pt",
            "cnn_attn":        "detector_cnn_attention_ft.pt",
            "cnn_bilstm_attn": "detector_cnn_bilstm_attn_ft.pt",
        }
        ckpt_path = MODEL_DIR / ckpt_map[model_name]
        print(f"[INFO] Loading detector from {ckpt_path}")

        if model_name == "cnn_only":
            self.detector = CNN_Only(feat_dim=FEAT_DIM, seq_len=SEQ_LEN)
        elif model_name == "cnn_attn":
            self.detector = CNN_Attention(feat_dim=FEAT_DIM, num_heads=4)
        elif model_name == "cnn_bilstm_attn":
            self.detector = CNN_BiLSTM_Attn(feat_dim=FEAT_DIM, num_heads=4)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.detector.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.detector.eval()
        for p in self.detector.parameters():
            p.requires_grad = False
        #the detector is always kept frozen during live operation
        #only the RL policy (Q-net) is updated online
        #we freeze the detector because it was carefully trained and fine-tuned
        #and we don't want live traffic to accidentally corrupt its weights

        # load RL policy — prefer existing online checkpoint (resume), else base
        base_ckpt_path   = Path(str(BASE_CKPT).replace("{model}", model_name))
        online_ckpt_path = Path(str(ONLINE_CKPT).replace("{model}", model_name))

        if online_ckpt_path.exists():
            print(f"[INFO] Resuming from online RL checkpoint: {online_ckpt_path}")
            rl_path_to_load = online_ckpt_path
        else:
            print(f"[INFO] Loading base RL policy from {base_ckpt_path}")
            rl_path_to_load = base_ckpt_path
        #if we previously ran this script and saved an online checkpoint we resume from it
        #this means the agent keeps all the learning it accumulated in past sessions
        #if no online checkpoint exists we start fresh from the base (accepted) policy

        self.q_net = QNet(STATE_DIM, N_ACTIONS)
        self.q_net.load_state_dict(torch.load(rl_path_to_load, map_location=self.device))
        self.q_net.eval()

        # online RL updater — pass both ckpt paths so drift rollback works
        self.online_updater = OnlineRLUpdater(
            self.q_net, self.device, online_ckpt_path, base_ckpt_path
        )
        #create the online updater and give it both checkpoint paths
        #online_ckpt_path is where it saves periodic updates
        #base_ckpt_path is the safe rollback target for the drift guard

        # sliding window + state tracker
        self.packet_buffer               = collections.deque(maxlen=SEQ_LEN)
        self.prev_time                   = None
        self.packets_since_last_decision = 0
        self.state_tracker               = LiveStateTracker()
        self.stats                       = LiveStats()
        self._decision_count             = [0]
        #packet_buffer holds the last 20 packets (deque automatically drops older ones)
        #prev_time is used to calculate time_delta between consecutive packets
        #packets_since_last_decision counts packets received since the last decision was made
        #state_tracker builds the 16-dim state vector and maintains fp/fn counters
        #stats collects all the numbers shown in the terminal dashboard

        # previous (state, action, det_p) for forming (s, a, r, s') tuples
        self._last_state  = None
        self._last_action = None
        self._last_det_p  = None
        #we need to remember the previous state and action because the reward for
        #an action is only computed at the NEXT step when we know the next state
        #this is the standard (s, a, r, s') tuple used in all RL algorithms

        print("[INFO] All models loaded  |  Online RL: ON")
        print(f"[INFO] Online checkpoint → {online_ckpt_path}")
        print(f"[INFO] Drift rollback to → {base_ckpt_path}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _scale_seq(self, seq: np.ndarray) -> np.ndarray:
        s = seq.copy()
        s[:, [0, 1, 2]] = self.scaler.transform(seq[:, [0, 1, 2]])
        return s
    #normalize the 3 continuous features (Time, time_delta, Length) using the real hardware scaler
    #we use .copy() so we don't modify the original packet buffer data

    def _detector_prob(self, seq: np.ndarray) -> float:
        seq_scaled = self._scale_seq(seq)
        x = torch.tensor(seq_scaled[None, ...], dtype=torch.float32)
        #[None, ...] adds a batch dimension so shape goes from (20,12) to (1,20,12)
        with torch.no_grad():
            p = torch.sigmoid(self.detector(x)).item()
        #sigmoid converts the raw model output to a probability between 0 and 1
        #torch.no_grad() skips gradient computation since the detector is frozen
        return float(p)
    #returns a single float like 0.87 meaning "87% probability this is an attack"

    def _pick_action(self, state: np.ndarray) -> int:
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        #unsqueeze(0) adds batch dimension: (16,) → (1, 16)
        with self.online_updater._lock:
            with torch.no_grad():
                q_vals = self.q_net(s_tensor)
                a      = int(torch.argmax(q_vals, dim=1).item())
        #we hold the lock while reading Q-values to prevent the update thread
        #from modifying Q-net weights at the same time we are doing inference
        #argmax picks the action with the highest Q-value (greedy policy)
        return a

    def _process_sequence(self):
        seq = np.array(list(self.packet_buffer), dtype=np.float32)
        #convert the deque of 20 packet feature vectors into a numpy array (20, 12)
        seq[:, 0] = seq[:, 0] - seq[:, 0].min()
        #make Time relative within this sequence (starts at 0) same as in training

        t_start = time.perf_counter()
        #start the clock to measure how long the inference takes

        det_p      = self._detector_prob(seq)
        #run the detector to get the attack probability for this 20-packet window
        seq_scaled = self._scale_seq(seq)
        state      = self.state_tracker.build_state(seq_scaled, det_p)
        #build the 16-dim state vector from the scaled sequence and detector probability
        action     = self._pick_action(state)
        #ask the RL agent which action to take given this state

        response_ms = (time.perf_counter() - t_start) * 1000.0
        #measure how long the whole inference process took in milliseconds
        action_name = ACTION_NAMES.get(action, "UNKNOWN")

        # form (s, a, r, s') tuple from the previous step
        if self._last_state is not None:
            reward = compute_live_reward(
                self._last_det_p,
                self._last_action,
                self.state_tracker.fp_counter,
                self.state_tracker.fn_counter,
                self.state_tracker.escalation_level,
            )
            self.online_updater.push(
                self._last_state,
                self._last_action,
                reward,
                state,
            )
        #we compute the reward for the PREVIOUS action now that we have the next state
        #this is the standard RL delayed feedback: we can only evaluate an action
        #after we see what state it led to
        #the experience tuple (last_state, last_action, reward, current_state) is pushed
        #to the online updater's pending queue

        # remember for next step
        self._last_state  = state.copy()
        self._last_action = action
        self._last_det_p  = det_p
        #save the current state, action and detector probability
        #so we can form the experience tuple at the NEXT step

        # update state tracker internals
        self.state_tracker.update(action, det_p)
        #update escalation_level, fp_counter and fn_counter based on this step

        self.stats.record_decision(det_p, action, action_name, response_ms)
        self._decision_count[0] += 1
        #record everything in the stats object for the dashboard
        #increment the shared decision counter so the update thread knows when to update

        # refresh online RL stats for dashboard
        self.stats.update_online_stats(
            self.online_updater.total_updates,
            self.online_updater.buffer_fill,
            self.online_updater.last_loss,
            self.online_updater.rollback_count,
        )
        #pull the latest online RL diagnostics into the stats object for the dashboard

    def _on_packet_struct(self, pkt):
        try:
            src_ip = pkt["src_ip"]
            sport  = pkt["sport"]
            dport  = pkt["dport"]
            flags  = pkt["flags"]
            length = pkt["length"]
            ts     = pkt["time"]
            #unpack all fields from the parsed packet dictionary

            self.stats.record_packet(src_ip)

            feat      = np.zeros(FEAT_DIM, dtype=np.float32)
            #create a zero array for all 12 features
            t_delta   = ts - self.prev_time if self.prev_time else 0.0
            self.prev_time = ts
            #calculate time gap since last packet (0 for the very first packet)

            feat[0]   = ts
            feat[1]   = t_delta
            feat[2]   = length
            to_mqtt   = int(dport in MQTT_PORTS)
            from_mqtt = int(sport in MQTT_PORTS)
            feat[3]   = int(to_mqtt or from_mqtt)
            feat[10]  = to_mqtt
            feat[11]  = from_mqtt
            feat[4]   = int(flags & 0x02 != 0)   # SYN
            feat[5]   = int(flags & 0x10 != 0)   # ACK
            feat[6]   = int(flags & 0x01 != 0)   # FIN
            feat[7]   = int(flags & 0x04 != 0)   # RST
            feat[8]   = int(flags & 0x08 != 0)   # PSH
            feat[9]   = int(flags & 0x20 != 0)   # URG
            #extract TCP flags using bitwise AND with the flag bitmask
            #0x02 is the SYN bit position in the TCP flags byte
            #0x10 is the ACK bit, 0x01 is FIN, 0x04 is RST, 0x08 is PSH, 0x20 is URG
            #int(condition) converts True/False to 1/0 for the feature vector

            self.packet_buffer.append(feat)
            #add this packet's features to the 20-packet sliding window buffer
            self.packets_since_last_decision += 1

            if len(self.packet_buffer) == SEQ_LEN and self.packets_since_last_decision >= STEP:
                self.packets_since_last_decision = 0
                self._process_sequence()
            #only trigger a decision when we have exactly 20 packets AND at least 5 new packets
            #have arrived since the last decision — this is the sliding window with step=5
        except Exception:
            pass
            #silently ignore malformed packets so one bad packet doesn't crash the whole system

    def _dashboard_thread(self):
        while self.running[0]:
            snap = self.stats.snapshot()
            render_dashboard(snap, self.model_name, self.iface)
            time.sleep(DISPLAY_REFRESH)
        #background thread that refreshes the terminal dashboard every 0.5 seconds
        #it runs independently of packet processing so the display stays updated even
        #when packet traffic is slow

    def parse_tshark_line(self, line):
        try:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                return None
            return {
                "time":   float(parts[0]),
                "src_ip": parts[1],
                "dst_ip": parts[2],
                "sport":  int(parts[3]),
                "dport":  int(parts[4]),
                "length": int(parts[5]),
                "flags":  int(parts[6], 16),
            }
        except Exception:
            return None
    #parse one line of tshark tab-separated output into a dictionary
    #tshark outputs fields separated by tabs in the order we specified with -e flags
    #parts[6] is the TCP flags field which tshark outputs as a hex string like "0x002"
    #int(parts[6], 16) converts from hex string to integer so we can use bitwise AND

    def run(self):
        print(">>> RUN FUNCTION STARTED", flush=True)
        self.running[0] = True

        threading.Thread(target=self._dashboard_thread, daemon=True).start()
        #start the dashboard refresh thread in the background
        #daemon=True means this thread will automatically stop when the main program exits

        threading.Thread(
            target=self.online_updater.run_update_loop,
            args=(self._decision_count, self.running),
            daemon=True,
        ).start()
        #start the online RL update thread in the background
        #it will monitor the decision counter and run gradient steps every 50 decisions

        print("[INFO] Starting remote tshark capture...")

        proc = subprocess.Popen(
            [
                "ssh",
                "-v",
                "-i", "/home/broker1/.ssh/id_rsa",
                "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "broker1@192.168.0.133",
                "sudo tshark -i wlan0 -l -T fields",
                "-e", "frame.time_epoch",
                "-e", "ip.src",
                "-e", "ip.dst",
                "-e", "tcp.srcport",
                "-e", "tcp.dstport",
                "-e", "frame.len",
                "-e", "tcp.flags",
                "-Y 'tcp.port == 1883 or tcp.port == 8883'"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        #we run tshark REMOTELY on the broker Raspberry Pi via SSH
        #the client2 Pi (where this script runs) cannot sniff the broker's traffic directly
        #so we SSH into broker1 and run tshark there then pipe the output back to us
        #-i wlan0 captures on the WiFi interface
        #-l means line-buffered output so we get packets immediately not in batches
        #-T fields means output tab-separated field values (not the full packet dump)
        #each -e flag adds one field to the output (time, IPs, ports, length, flags)
        #-Y 'tcp.port == 1883 or tcp.port == 8883' is the capture filter
        #stdout=subprocess.PIPE means we can read tshark's output line by line in Python
        #text=True means output comes as strings not bytes
        #bufsize=1 means line-buffered so each line arrives as soon as tshark outputs it

        print(">>> SSH PROCESS STARTED", flush=True)

        def read_err(pipe):
            for line in pipe:
                print("ERR: ", line, flush=True)

        threading.Thread(target=read_err, args=(proc.stderr,), daemon=True).start()
        #read SSH/tshark error messages in a background thread so they don't block packet reading
        #this helps us see connection errors or permission denied messages for debugging

        if proc.poll() is not None:
            print("SSH PROCESS DIED. RETURN CODE:", proc.returncode, flush=True)
            self.stop()
        #check if the SSH process immediately exited (connection refused, wrong key, etc.)
        #poll() returns None if the process is still running or the return code if it exited

        def print_packets(proc):
            try:
                for line in proc.stdout:
                    print("raw: ", line.strip(), flush=True)
                    pkt = self.parse_tshark_line(line)
                    if pkt:
                        print("parsed: ", pkt, flush=True)
                        self._on_packet_struct(pkt)
            except KeyboardInterrupt:
                self.stop()
        #read each line from tshark output as it arrives
        #parse it into a packet dictionary and pass it to _on_packet_struct for processing
        #we print both the raw line and the parsed dictionary for debugging purposes

        print_packets(proc)

    def stop(self):
        self.running[0] = False
        online_ckpt = Path(str(ONLINE_CKPT).replace("{model}", self.model_name))
        with self.online_updater._lock:
            torch.save(self.q_net.state_dict(), online_ckpt)
        print(f"\n[INFO] Online RL checkpoint saved → {online_ckpt}")
        print(
            f"[INFO] NeuroGuard v2 stopped.  "
            f"Packets: {self.stats.total_packets:,}  "
            f"Decisions: {self.stats.total_decisions:,}  "
            f"Online updates: {self.online_updater.total_updates}  "
            f"Rollbacks: {self.online_updater.rollback_count}"
        )
        #stop() is called when Ctrl+C is pressed or when the SSH process dies
        #setting running[0]=False signals all background threads to exit their loops
        #we save the current Q-net weights under the lock so the update thread
        #doesn't write at the same time
        #we print a final summary of everything that happened during this session


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeuroGuard Live v2 — Online RL Adaptation")
    parser.add_argument(
        "--model", default="cnn_bilstm_attn",
        choices=["cnn_only", "cnn_attn", "cnn_bilstm_attn"],
    )
    parser.add_argument("--iface", default=DEFAULT_IFACE)
    args = parser.parse_args()
    #argparse reads command-line arguments so we can run:
    #python neuroguard_live_v2_online_rl.py --model cnn_bilstm_attn
    #--model lets us choose which of the three detector architectures to use
    #default is cnn_bilstm_attn which is our best-performing model
    #--iface lets us specify a different network interface if needed (default wlan0)

    defender = None

    def _handle_ctrl_c(sig, frame):
        print(f"\n\n{YELLOW}[INFO] Ctrl+C — stopping and saving...{RESET}")
        if defender:
            defender.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_ctrl_c)
    signal.signal(signal.SIGTERM, _handle_ctrl_c)
    #register Ctrl+C (SIGINT) and kill signal (SIGTERM) handlers
    #so when the user presses Ctrl+C the model is saved cleanly before the program exits
    #without this the program would exit immediately and we would lose any unsaved online learning

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"  NeuroGuard Live Defender  —  VERSION 2")
    print(f"  Model      : {args.model}")
    print(f"  Model dir  : {MODEL_DIR}")
    print(f"  RL update  : every {ONLINE_UPDATE_EVERY} decisions  |  LR={ONLINE_LR}")
    print(f"  Drift guard: ALLOW floor={ALLOW_FLOOR*100:.0f}%  window={DRIFT_WINDOW}")
    print(f"{BOLD}{'═'*60}{RESET}\n")

    if not MODEL_DIR.exists():
        print(f"{RED}[ERROR] Model directory not found: {MODEL_DIR}{RESET}")
        sys.exit(1)
    #check that the model folder exists on the Pi before doing anything
    #if it doesn't exist we print a clear error so the user knows to create it or copy models

    defender = NeuroGuardLive(model_name=args.model, iface=args.iface)
    defender.run()
    #create the main defender object (loads all models) and start capturing packets


if __name__ == "__main__":
    main()
