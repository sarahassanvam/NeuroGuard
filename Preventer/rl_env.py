# no runn
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
import pickle
import math

# the list of actions
A_ALLOW = 0  # means do nothing, let traffic pass
A_RATE_LIMIT = 1  # means to limit request rate it's a light mitigation
A_TEMP_BLOCK = 2  # means temporarily block ip/client
A_PERM_BLOCK = 3  # means permanently block ip/client
A_DROP_SYN = 4  # means to drop tcp syn packets
A_DROP_CONNECT = 5  # means to drop connection attempts
A_DELAY_CONNECT = 6  # means to delay connection attempts
A_LIMIT_PUBLISH = 7  # this to limit mqtt publish messages
A_BLOCK_SUBSCRIBE = 8  # to block MQTT subscribe messages
A_DISCONNECT = 9  # this is to force client disconnect
A_QUARANTINE = 10  # to isolate client from normal network
A_ISOLATE_NODE = 11  # fully isolate node
A_REDUCE_QOS = 12  # means to reduce mqtt quality of service
A_ALERT_ONLY = 13  # aims to only raising alert so no mitigation
A_ESCALATE = 14  # increase response severity
A_DEESCALATE = 15  # decrease response severity
#physical layer: ISOLATE_NODE and QUARANTINE_CLIENT
#network (ip) layer: RATE_LIMIT_IP and TEMP_BLOCK_IP and PERM_BLOCK_IP
#transport layer: DROP_CONNECT and DELAY_CONNECT and DISCONNECT_CLIENT and DROP_SYN_DELAY_TCP
#application (mqtt) layer: LIMIT_PUBLISH and BLOCK_SUBSCRIBE and REDUCE_QOS
#policy: ALLOW and ALERT_ONLY and ESCALATE and DEESCALATE

ACTION_NAMES = {
    0: "ALLOW",  # it means that it's in human readable name for action 0 and so on for all the actions
    1: "RATE_LIMIT_IP",
    2: "TEMP_BLOCK_IP",
    3: "PERM_BLOCK_IP",
    4: "DROP_SYN_DELAY_TCP",
    5: "DROP_CONNECT",
    6: "DELAY_CONNECT",
    7: "LIMIT_PUBLISH",
    8: "BLOCK_SUBSCRIBE",
    9: "DISCONNECT_CLIENT",
    10: "QUARANTINE_CLIENT",
    11: "ISOLATE_NODE",
    12: "REDUCE_QOS",
    13: "ALERT_ONLY",
    14: "ESCALATE",
    15: "DEESCALATE",
    # in short it maps action id to readable string
}

# what counts as safe or normal-friendly actions
# in other words actions that are safe on normal traffic
SAFE_ACTIONS = {A_ALLOW, A_ALERT_ONLY, A_DEESCALATE}
# in a strong actions it means that it can break normal users
HEAVY_ACTIONS = {A_TEMP_BLOCK, A_PERM_BLOCK, A_DISCONNECT, A_QUARANTINE, A_ISOLATE_NODE}
# actions that do not directly stop an attack
NON_MITIGATE_ACTIONS = {A_ALLOW, A_ALERT_ONLY, A_ESCALATE, A_DEESCALATE}


def add_features(df: pd.DataFrame):
    # this creates the feature columns used by the model
    # we rebuild them after splitting so that preprocessing stays consistent
    feat = df.copy()

    feat["Time"] = pd.to_numeric(feat["Time"], errors="coerce").fillna(0.0)
    feat["Length"] = pd.to_numeric(feat["Length"], errors="coerce").fillna(0.0)
    # force numeric and fill missing to avoid nans breaking training

    # normalize Time per file so model doesn't depend on absolute capture start
    feat["Time"] = feat.groupby("file_name")["Time"].transform(lambda s: s - s.min())

    feat["time_delta"] = feat.groupby("file_name")["Time"].diff().fillna(0.0)
    # time difference between consecutive packets inside the same file

    info = feat["Info"].astype(str).str.upper()
    # now normalizing the info text so keyword matching is consistent

    # extract source and destination ports from pattern:
    # 1883 to 53415
    # 51781 to 1883
    ports = info.str.extract(r"(\d+)\s*→\s*(\d+)")
    sport = pd.to_numeric(ports[0], errors="coerce")
    dport = pd.to_numeric(ports[1], errors="coerce")

    mqtt_ports = {1883, 8883}

    # mqtt port presence
    feat["has_mqtt_port"] = ((sport.isin(mqtt_ports)) | (dport.isin(mqtt_ports))).fillna(False).astype(int)

    # direction features
    feat["to_mqtt"] = (dport.isin(mqtt_ports)).fillna(False).astype(int)
    feat["from_mqtt"] = (sport.isin(mqtt_ports)).fillna(False).astype(int)

    # tcp flags
    feat["flag_syn"] = info.str.contains(r"\bSYN\b", regex=True, na=False).astype(int)
    feat["flag_ack"] = info.str.contains(r"\bACK\b", regex=True, na=False).astype(int)
    feat["flag_fin"] = info.str.contains(r"\bFIN\b", regex=True, na=False).astype(int)
    feat["flag_rst"] = info.str.contains(r"\bRST\b", regex=True, na=False).astype(int)
    feat["flag_psh"] = info.str.contains(r"\bPSH\b", regex=True, na=False).astype(int)
    feat["flag_urg"] = info.str.contains(r"\bURG\b", regex=True, na=False).astype(int)

    # final list of 12 features
    feature_cols = [
        "Time", "time_delta", "Length",
        "has_mqtt_port",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
        "to_mqtt", "from_mqtt",
    ]
    # 3 continuous + 9 binary = 12 features total

    return feat, feature_cols


def build_sequences_per_file(
        # this function is to create fixed length sequences without mixing files
        df_feat: pd.DataFrame,
        feature_cols: List[str],
        seq_len=20,
        step=5
):
    X_list, y_list, file_list = [], [], []

    for fname, g in df_feat.groupby("file_name", sort=False):
        # this means process one capture file at a time
        g = g.sort_values("Time")
        # then ensure correct time order
        X = g[feature_cols].to_numpy(dtype=np.float32)
        # extract feature matrix from our detector
        y = g["label"].astype(int).to_numpy()
        # then extract the labels they are used ONLY for reward

        # now we are writing a base rule just in case and it means skip the files that are too short to form a sequence
        if len(X) < seq_len:
            continue

        for start in range(0, len(X) - seq_len + 1, step):
            # this means sliding window over packets
            # it means start at packet 0
            # then move forward by step packets
            # then stop when there aren't enough packets left to form a full window
            end = start + seq_len  # this is to decide where the window ends
            X_list.append(X[start:end])  # it means take packets from start to end
            # then store them as one training example
            y_list.append(int(y[start:end].max()))
            # now this means look at labels of packets in this window
            # if any packet is an attack it means the whole window is attack
            file_list.append(str(fname))
            # to track which file this sequence came from

    if len(X_list) == 0:
        raise RuntimeError("No sequences created, reduce SEQ_LEN or check data..")
    # the base rule as always need to be here
    X = np.stack(X_list).astype(np.float32)  # this means combine all sequences into one big array
    y = np.array(y_list, dtype=np.int64)  # this means to convert sequence labels into a numpy array so 1 label per seq
    files = np.array(file_list, dtype=object)  # this to store file name for each sequence
    # and it will be used later to group sequences into episodes
    return X, y, files


#MultiHeadAttention instead of simple AttentionPooling
class MultiHeadAttention(nn.Module):
    # multi-head attention mechanism to focus on important packets from different perspectives
    # this helps the model learn multiple patterns simultaneously
    # for example: one head might focus on timing, another on message types
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        # hidden_dim is the size of the feature vector for each packet after cnn processing
        # num_heads is how many different attention patterns we want to learn
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # split hidden_dim across heads so each head processes head_dim features

        # query, key, value projections for all heads (but done in one matrix multiply for efficiency)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # query: what am I looking for
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # key: what information do I contain
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # value: what information do I provide

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # final projection to combine all heads

        self.scale = math.sqrt(self.head_dim)
        # scaling factor to prevent dot products from getting too large

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        # this means we have a batch of sequences
        # each sequence has seq_len packets
        # each packet has hidden_dim features
        B, T, D = x.shape

        # compute Q, K, V for all heads
        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)  # (B, T, D)
        V = self.v_proj(x)  # (B, T, D)

        # reshape to separate heads (B, T, D) to (B, T, num_heads, head_dim) to (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        #scaled dot-product attention
        #compute how much each packet should attend to every other packet
        scores = (Q @ K.transpose(-2, -1)) / self.scale  # (B, num_heads, T, T)
        #divide by scale to prevent gradients from vanishing

        attn_weights = torch.softmax(scores, dim=-1)
        #convert scores to probabilities that sum to 1
        #softmax ensures all weights are positive and add up to 1

        #apply attention weights to values
        out = attn_weights @ V  #(B, num_heads, T, head_dim)
        #multiply attention weights by values to get weighted combination

        #concatenate heads (B, num_heads, T, head_dim) to (B, T, num_heads, head_dim) to (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # final projection
        out = self.out_proj(out)  # (B, T, D)

        #pool over time dimension by averaging
        out = out.mean(dim=1)  # (B, D)
        #average across all packets to get one vector per sequence

        return out


class CNN_Only(torch.nn.Module):
    def __init__(self, feat_dim, seq_len=20):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * seq_len, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.fc(x).squeeze(1)


class CNN_Attention(torch.nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128)
        )
        self.multi_head_attn = MultiHeadAttention(128, num_heads=num_heads)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.multi_head_attn(x)
        return self.fc(x).squeeze(1)


class CNN_BiLSTM_Attn(torch.nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128)
        )
        self.bilstm = torch.nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.multi_head_attn = MultiHeadAttention(128, num_heads=num_heads)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = self.multi_head_attn(x)
        return self.fc(x).squeeze(1)


# now this class defines the reinforcement learning environment
class NeuroGuardRLEnv:

    def __init__(  # initializing the rl environment
            self,  # self is the reference to this environment object
            # now the input sequences we have the number of sequences * packet * feature of the packet
            X_seq: np.ndarray,  # (Nseq, T, F)
            # and the labels per sequence (used only for reward)
            y_seq: np.ndarray,  # (Nseq,)
            # and then the file name for each sequence (defines episode boundaries)
            # now an episode means all traffic from one capture file (file_name not the ones from my original dataset) processed from start to end
            files_seq: np.ndarray,  # (Nseq,)
            # this is the path to our pretrained detector model weights
            detector_ckpt: Path,
            # this is the path to the scaler used during detector training
            scaler_ckpt: Path,
            # this is the device where detector runs (cpu i did)
            device: str = "cpu",
            # detector type: "cnn_only", "cnn_attention", or "cnn_bilstm_attn"
            detector_type: str = "cnn_bilstm_attn",
            # it like the base rule as a safety limit to stop very long episodes
            max_steps_per_episode: int = 300,
            # optional cost for each action (penalize heavy actions)
            action_costs: Dict[int, float] | None = None,
            # this is the detector probability threshold to consider traffic as attack
            # we choose 0.5 because sigmoid outputs a probability
            # and 0.5 is the natural middle decision boundary
            det_attack_thr: float = 0.5,
            # and this is the detector's confidence of which traffic is strongly normal
            det_normal_low_thr: float = 0.4,
    ):
        self.X_seq = X_seq  # then storing the feature sequences
        self.y_seq = y_seq  # the labels
        self.files_seq = files_seq  # file association per sequence
        self.detector_type = detector_type  #store detector type for adaptive rewards
        self.device = device  # then saving the device setting
        self.det_attack_thr = float(det_attack_thr)  # and the attack threshold
        self.det_normal_low_thr = float(det_normal_low_thr)  # and the normal traffic threshold

        # load the scaler
        with open(scaler_ckpt, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"[INFO] Loaded scaler from: {scaler_ckpt}")

        # create the correct detector model based on type
        feat_dim = X_seq.shape[-1]
        if detector_type == "cnn_only":
            self.detector = CNN_Only(feat_dim=feat_dim, seq_len=X_seq.shape[1]).to(device)
        elif detector_type == "cnn_attention":
            self.detector = CNN_Attention(feat_dim=feat_dim, num_heads=4).to(device)
        elif detector_type == "cnn_bilstm_attn":
            self.detector = CNN_BiLSTM_Attn(feat_dim=feat_dim, num_heads=4).to(device)
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        print(f"[INFO] Created detector: {detector_type} with {feat_dim} features")

        # now we will create the detector model with correct feature size
        self.detector.load_state_dict(torch.load(detector_ckpt, map_location=device))
        # then we will load the saved best weigghts
        self.detector.eval()
        # now this is uesd cuz the detector is pretrained and the rl
        # must not change or randomize it so we force inference mode
        # and this for freezing the detector so rl doesn't change it
        for p in self.detector.parameters():
            p.requires_grad = False  # requires_grad=False means don't learn or update this parameter

        self.max_steps_per_episode = int(max_steps_per_episode)
        # then storing the maximum episode length
        # now i didn't provide action_costs when creating the environment
        # so the code inside this block will run(to be clearler)
        # make heavy actions expensive so the agent doesn't block everything
        if action_costs is None:
            action_costs = {a: 0.0 for a in ACTION_NAMES.keys()}  # the default cost=0 for all actions

            #cnn-only gets cheaper actions
            if detector_type == "cnn_only":
                #reduced costs to encourage exploration
                action_costs.update({
                    A_RATE_LIMIT: 0.10,
                    A_DROP_SYN: 0.15,
                    A_DROP_CONNECT: 0.20,
                    A_DELAY_CONNECT: 0.15,
                    A_LIMIT_PUBLISH: 0.15,
                    A_BLOCK_SUBSCRIBE: 0.15,
                    A_REDUCE_QOS: 0.15,
                    A_ALERT_ONLY: 0.03,
                    A_ESCALATE: 0.30,
                    A_DEESCALATE: 0.05,
                    A_TEMP_BLOCK: 0.40,
                    A_PERM_BLOCK: 0.70,
                    A_DISCONNECT: 0.60,
                    A_QUARANTINE: 0.70,
                    A_ISOLATE_NODE: 0.90,
                })
            else:
                #CNN-attention and CNN-BiLSTM-attention will use the standard costs
                action_costs.update({
                    A_RATE_LIMIT: 0.15,
                    A_DROP_SYN: 0.20,
                    A_DROP_CONNECT: 0.30,
                    A_DELAY_CONNECT: 0.20,
                    A_LIMIT_PUBLISH: 0.25,
                    A_BLOCK_SUBSCRIBE: 0.25,
                    A_REDUCE_QOS: 0.20,
                    A_ALERT_ONLY: 0.05,
                    A_ESCALATE: 0.40,
                    A_DEESCALATE: 0.10,
                    A_TEMP_BLOCK: 0.60,
                    A_PERM_BLOCK: 1.00,
                    A_DISCONNECT: 0.80,
                    A_QUARANTINE: 1.00,
                    A_ISOLATE_NODE: 1.20,
                })

        self.action_costs = action_costs  # and finally store action cost table
        # we gave penalties to each action so the agent learns to avoid
        # them unless they're really needed like in the case of serious attacks when they are 100% sure
        # this helps it to make smarter decisions

        # self.file_to_indices it's a dictionary where we have a key and value
        # so it maps each file name to the sequence numbers (positions) within that file
        self.file_to_indices: Dict[str, List[int]] = {}
        for i, fn in enumerate(self.files_seq):  # this will loop over all sequences
            key = str(fn)  # to convert file name to string
            self.file_to_indices.setdefault(key, []).append(i)  # then group the sequence index under its file
        self.file_names = list(self.file_to_indices.keys())
        # extracting the list of file names (the keys) from the dictionary
        # it will be useful when resetting the environment for example picking the next file for processing

        ##this is the currently active file (episode)
        self.cur_file: str | None = None
        self.cur_indices: List[int] | None = None  # this is the sequence indices for the current episode(file)
        self.t = 0  # this tracks the current sequence (time step) in the current episode (file)
        self.steps = 0  # this is the global step counter

        self.prev_action = A_ALLOW  # says the previous action taken by agent
        self.escalation_level = 0  # the current escalation level
        self.fp_counter = 0  # and a false positive counter (blocking normal)
        self.fn_counter = 0  # and a false negative counter (missing attacks)

    def _scale_seq(self, seq_x: np.ndarray) -> np.ndarray:
        # apply the same scaling used during training
        continuous_idx = [0, 1, 2]  # Time, time_delta, Length
        seq_x_scaled = seq_x.copy()
        seq_x_scaled[:, continuous_idx] = self.scaler.transform(seq_x[:, continuous_idx])
        return seq_x_scaled

    # this function will compute the detector's attack probability for one sequence
    def _detector_prob(self, seq_x: np.ndarray) -> float:
        seq_x_scaled = self._scale_seq(seq_x)

        xb = torch.tensor(seq_x_scaled[None, ...], dtype=torch.float32, device=self.device)
        # the model expects data in batches so we add a batch dimension
        # even if it's just one sequence
        with torch.no_grad():  # tells PyTorch not to calculate gradients since we are not training just inferring
            # now pass the sequence (xb) through the detector model to get the logits (raw prediction values)
            logit = self.detector(xb)
            p = torch.sigmoid(logit).item()  # now convert the logits to a probability using the sigmoid function
        return float(p)  # this returns detector probability

    def _make_state(self, seq_x: np.ndarray, det_p: float) -> np.ndarray:
        # here we will build the rl state vector
        # the state vector is used by the agent to understand its current situation
        # (based on the packets and detection) and then decides on an action (for example allow, block, alert)

        #use scaled continuous features so state matches detector training distribution
        seq_x_scaled = self._scale_seq(seq_x)

        time_delta_mean = float(np.mean(seq_x_scaled[:, 1]))  # the average time between consecutive packets in a sequence
        length_mean = float(np.mean(seq_x_scaled[:, 2]))  # the average packet size
        has_mqtt_port_mean = float(np.mean(seq_x_scaled[:, 3]))  # the fraction of packets that touch mqtt port (1883/8883)

        # flags mean (8 values): syn ack fin rst psh urg to_mqtt from_mqtt
        flags_mean = np.mean(seq_x_scaled[:, 4:12], axis=0).astype(np.float32)

        extras = np.array([  # this will aim to build extra state features
            det_p,  # detector probability
            time_delta_mean,
            length_mean,
            has_mqtt_port_mean,
            float(self.prev_action) / 15.0,  # previous action
            float(self.escalation_level) / 5.0,  # escalation level
            float(self.fp_counter) / 10.0,  # false positive pressure
            float(self.fn_counter) / 10.0,  # false negative pressure
        ], dtype=np.float32)  # so in total of 16 features
        # and dividing by 5 and 10 normalizes the counts to keep them in a manageable range

        return np.concatenate([extras, flags_mean], axis=0).astype(np.float32)
        # the final state vector (16 values)

    def reset(self, file_name: str | None = None) -> np.ndarray:
        # this means start a new episode
        if file_name is None:  # and randomly pick a file if none specified
            self.cur_file = str(np.random.choice(self.file_names))
        else:
            self.cur_file = str(file_name)

        self.cur_indices = self.file_to_indices[self.cur_file]
        # this aims to load sequence indices for this episode

        self.t = 0
        self.steps = 0
        # then reset the counters
        self.prev_action = A_ALLOW
        self.escalation_level = 0
        self.fp_counter = 0
        self.fn_counter = 0
        # as well as resetting the internal state

        idx = self.cur_indices[self.t]  # as before the first sequence index
        seq_x = self.X_seq[idx]  # then get the feature sequence
        det_p = self._detector_prob(seq_x)  # the computing the detector's probability
        # cuz we want to know if it's an attack or not to take an actiom
        return self._make_state(seq_x, det_p)  # then return the initial state

    # now we will apply one rl action
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        action = int(action)  # first to make sure the action is integer
        if self.cur_indices is None:
            raise RuntimeError("Call reset() before step().")
        # this is like the base rule to ensure that reset() is called first
        # to set up the environment before calling step()

        idx = self.cur_indices[self.t]  # again the current sequence index
        seq_x = self.X_seq[idx]  # the feature of that seq
        y_true = int(self.y_seq[idx])  # labels only for reward

        det_p = self._detector_prob(seq_x)  # the detector probability
        det_attack = det_p >= self.det_attack_thr  # we will check if the detector's
        # probability is above the threshold to classify as an attack

        reward = 0.0  # initializing the reward as zero

        #cnn-only gets more forgiving rewards
        is_cnn_only = (self.detector_type == "cnn_only")

        # attack traffic reward only real mitigation
        if y_true == 1:  # first we will check if the traffic is an attack
            if action == A_ALLOW:  # we will give penalty for allowing an attack to pass without mitigation
                reward -= 2.0 if is_cnn_only else 3.0  # REDUCED for CNN-only
                self.fn_counter = min(self.fn_counter + (1 if is_cnn_only else 2), 10)  # REDUCED counter increment

            elif action in NON_MITIGATE_ACTIONS:
                # ESCALATE/DEESCALATE/ALERT_ONLY is not real mitigation during attack
                reward -= 0.5 if is_cnn_only else 0.8  # REDUCED for CNN-only
                self.fn_counter = min(self.fn_counter + 1, 10)

            else:
                # if a real mitigation was taken then we will give it a reward
                reward += 1.5 if is_cnn_only else 1.2  # INCREASED for CNN-only
                # we will give bonus for matching the detector's prediction, encouraging consistency
                if det_attack:
                    reward += 0.5 if is_cnn_only else 0.3  # INCREASED for CNN-only

                # and a penalty will be given for jumping straight to heavy actions
                # like blocking without a need
                if action in HEAVY_ACTIONS:
                    reward -= 0.1 if is_cnn_only else 0.2  # REDUCED for CNN-only

                self.fn_counter = max(self.fn_counter - 1, 0)

        # normal traffic we prefer prefer ALLOW
        else:
            if action == A_ALLOW:
                reward += 2.5 if is_cnn_only else 2.0  # INCREASED for CNN-only
                self.fp_counter = max(self.fp_counter - 2, 0)

            elif action in SAFE_ACTIONS:
                # ALERT_ONLY or DEESCALATE then they are acceptable
                reward += 0.8 if is_cnn_only else 0.6  # INCREASED for CNN-only
                self.fp_counter = max(self.fp_counter - 1, 0)

            else:
                # blocking normal will have a penalty
                reward -= 2.0 if is_cnn_only else 2.5  # REDUCED for CNN-only
                self.fp_counter = min(self.fp_counter + 2, 10)

            # to decrese the tn so we will add this
            # so if the detector confidence is low then punish non allow more
            if det_p < self.det_normal_low_thr and action != A_ALLOW:
                reward -= 1.0 if is_cnn_only else 1.5  # REDUCED for CNN-only

            # explicitly punish ESCALATE on normal to prevent ESCALATE spamming
            # before fixing it it was not the best
            if action == A_ESCALATE:
                reward -= 2.0 if is_cnn_only else 2.5  # REDUCED for CNN-only

        # it subtracts penalties for actions from the reward
        # so the agent learns to take better (cheaper) actions
        reward -= float(self.action_costs.get(action, 0.0))
        # the 0.0 means that if an action has no cost defined in the dictionary
        # it gets a default cost of zero so it doesn't subtract anything from the reward

        # A_ESCALATE is an action that increases the escalation level,
        # meaning the agent takes more aggressive actions like blocking
        if action == A_ESCALATE:
            self.escalation_level = min(self.escalation_level + 1, 5)
            # the min() function ensures that the escalation level never goes above 5
        elif action == A_DEESCALATE:
            self.escalation_level = max(self.escalation_level - 1, 0)
            # the max() function ensures the escalation level never goes below 0
            # meaning it can't de-escalate below the lowest level

        # this line keeps track of the last action the agent took the agent needs
        # this to remember what it did previously which can influence its future decisions
        self.prev_action = action

        # this mean to move to the next sequence in the file
        # and the next will count one more action that has been taken
        self.t += 1
        self.steps += 1

        done = (self.t >= len(self.cur_indices)) or (self.steps >= self.max_steps_per_episode)
        # this means to stop if:
        # there is no more sequences in the file or the maximum allowed steps have been reached
        # it means that episode ended so it means there is no next state so return zeros as a default value
        if done:
            next_state = np.zeros(16, dtype=np.float32)
        else:
            next_idx = self.cur_indices[self.t]
            next_seq_x = self.X_seq[next_idx]
            next_p = self._detector_prob(next_seq_x)
            next_state = self._make_state(next_seq_x, next_p)
            # here it means that if episode continues
            # then get the next sequence
            # compute detector probability
            # then build the next state for the agent then make the action

        info = {
            "file": self.cur_file,
            "det_p": float(det_p),
            "y_true": int(y_true),
            "action": int(action),
            "action_name": ACTION_NAMES.get(int(action), "UNKNOWN"),
            "is_safe_action": bool(action in SAFE_ACTIONS),
            "is_heavy_action": bool(action in HEAVY_ACTIONS),
            # now we will store useful details
            # file name, detector probability, true label, chosen action & whether action was safe or heavy
        }
        return next_state, float(reward), bool(done), info  # and now give the agent:
        # next state, reward, done flag & extra info


# now this function prepares train / validation / test data for rl without data leakage

def build_rl_data_from_csv(
        csv_all: Path,
        train_idx: Path,
        val_idx: Path,
        test_idx: Path,
        seq_len: int = 20,
        step: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    # now this is a function that returns a dictionary
    # and inside it are train/val/test
    # each one contains NumPy arrays which are the (X, y, files)
    full = pd.read_csv(csv_all)  # this will read the entire csv file with all packets

    # these will load which rows belong to train / val / test
    tr = np.load(train_idx)
    va = np.load(val_idx)
    te = np.load(test_idx)
    # and then we will create three separate datasets using those indices
    train_df = full.iloc[tr].copy()
    val_df = full.iloc[va].copy()
    test_df = full.iloc[te].copy()

    # then we'll create numerical features for all the splits
    train_feat, feature_cols = add_features(train_df)
    val_feat, _ = add_features(val_df)
    test_feat, _ = add_features(test_df)
    # now turn packets into sequences (windows)
    # then keep sequences inside the same file
    Xtr, ytr, ftr = build_sequences_per_file(train_feat, feature_cols, seq_len, step)
    Xva, yva, fva = build_sequences_per_file(val_feat, feature_cols, seq_len, step)
    Xte, yte, fte = build_sequences_per_file(test_feat, feature_cols, seq_len, step)
    # X is the sequences and y the labels and the files are the file name per sequence
    return {
        "train": {"X": Xtr, "y": ytr, "files": ftr},
        "val": {"X": Xva, "y": yva, "files": fva},
        "test": {"X": Xte, "y": yte, "files": fte},
        "feature_cols": {"cols": np.array(feature_cols, dtype=object)},
    }
