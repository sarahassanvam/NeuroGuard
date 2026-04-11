import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

OUT_DIR   = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")
PCAP_FILE = Path(r"C:\Users\User\Downloads\broker_data_transfer.csv")
#OUT_DIR is where all my trained model files are saved from our earlier training scripts
#PCAP_FILE is the new real hardware CSV we captured from our actual Raspberry Pi broker
#this file contains real network traffic so we will use it to adapt our models to real conditions
SEQ_LEN    = 20
STEP       = 5
FEAT_DIM   = 12
MQTT_PORTS = {1883, 8883}
FT_EPOCHS  = 10
FT_LR      = 1e-4
FT_BATCH   = 32
EVAL_CHUNK = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
#SEQ_LEN = 20 means we group every 20 packets together as one sequence (same as training)
#STEP = 5 means we slide the window by 5 packets each time to create overlapping sequences
#FEAT_DIM = 12 means each packet is described by 12 features (same 12 we always used)
#MQTT_PORTS = the two standard MQTT ports we look for (1883 is plain, 8883 is secure)
#FT_EPOCHS = 10 means we run 10 full passes over the real hardware data during fine-tuning
#FT_LR = 1e-4 is a very small learning rate so fine-tuning gently updates the model
#without destroying what it already learned from the big training dataset
#FT_BATCH = 32 means we process 32 sequences at a time in each gradient step
#EVAL_CHUNK = 512 means when we check accuracy we process 512 sequences at once
#to avoid running out of memory
#DEVICE will use the GPU if available otherwise it falls back to CPU
DETECTOR_CONFIGS = {
    "cnn_only": {
        "ckpt_in":    OUT_DIR / "detector_cnn_only.pt",
        "scaler_in":  OUT_DIR / "scaler_cnn_only.pkl",
        "type":       "cnn_only",
        "ckpt_out":   OUT_DIR / "detector_cnn_only_ft.pt",
        "scaler_out": OUT_DIR / "scaler_cnn_only_ft.pkl",
    },
    "cnn_attn": {
        "ckpt_in":    OUT_DIR / "detector_cnn_attention.pt",
        "scaler_in":  OUT_DIR / "scaler_cnn_attention.pkl",
        "type":       "cnn_attention",
        "ckpt_out":   OUT_DIR / "detector_cnn_attention_ft.pt",
        "scaler_out": OUT_DIR / "scaler_cnn_attention_ft.pkl",
    },
    "cnn_bilstm_attn": {
        "ckpt_in":    OUT_DIR / "detector_cnn_bilstm_attn.pt",
        "scaler_in":  OUT_DIR / "scaler_cnn_bilstm_attn.pkl",
        "type":       "cnn_bilstm_attn",
        "ckpt_out":   OUT_DIR / "detector_cnn_bilstm_attn_ft.pt",
        "scaler_out": OUT_DIR / "scaler_cnn_bilstm_attn_ft.pkl",
    },
}
#this dictionary holds the file paths for each of our three detector models
#ckpt_in is the original model file we trained earlier (the starting weights)
#scaler_in is the original normalizer that was fit on the simulation dataset
#type tells us which architecture this entry belongs to
#ckpt_out is where we will save the new fine-tuned model weights after adapting to real hardware
#scaler_out is where we will save the new normalizer that is fit on real hardware traffic
#we have three detectors: CNN only, CNN with Attention, and CNN with BiLSTM and Attention
#all three will be fine-tuned using the same real hardware data
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        #we check that hidden_dim can be divided evenly by num_heads
        #if this fails it means the head sizes would be unequal which breaks the math
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        #head_dim is how big each attention head is so if hidden_dim=128 and num_heads=4
        #then each head looks at 32 dimensions of information independently
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        #q_proj, k_proj and v_proj are the three learnable projections
        #Query asks "what am I looking for?"
        #Key says "what information do I have?"
        #Value says "what information should I share?"
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        #out_proj combines the outputs from all heads back into one vector
        self.scale     = math.sqrt(self.head_dim)
        #we divide attention scores by the square root of head_dim to stop the scores from
        #getting too large which would make softmax output very sharp and kill gradients
    def forward(self, x):
        B, T, D = x.shape
        #B is batch size (how many sequences), T is time steps (20 packets), D is dimension (128)
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        #we project then reshape so each head has its own slice of the data
        #transpose(1,2) puts the head dimension before the time dimension for easier math
        scores       = (Q @ K.transpose(-2, -1)) / self.scale
        #dot product between Q and K tells us how much each packet should attend to each other
        #dividing by scale prevents the values from getting too large
        attn_weights = torch.softmax(scores, dim=-1)
        #softmax turns the scores into probabilities that sum to 1
        #this tells the model how much attention to pay to each packet in the sequence
        out          = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, D)
        #multiply attention weights by Values to get the weighted output
        #then reshape back to the original (B, T, D) format
        return self.out_proj(out).mean(dim=1)
        #out_proj mixes the head outputs together
        #mean(dim=1) collapses the time dimension so we get one vector per sequence

class CNN_Only(nn.Module):
    def __init__(self, feat_dim=12, seq_len=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        #the conv part is two 1D convolutional layers that slide across the 20-packet window
        #kernel_size=3 means each filter looks at 3 packets at a time to find local patterns
        #padding=1 keeps the output length the same as the input so we don't lose edge packets
        #128 filters means the model learns 128 different pattern detectors
        #ReLU keeps only positive activations (removes negatives) to add non-linearity
        #BatchNorm normalizes the layer outputs so training is more stable and faster
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * seq_len, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        #Flatten turns the 2D output of conv (128 filters x 20 time steps) into one long vector
        #the first Linear layer compresses it from 128*20=2560 down to 128 neurons
        #Dropout(0.3) randomly turns off 30% of neurons during training so the model
        #doesn't memorize the training data (prevents overfitting)
        #the last Linear(128,1) outputs a single score (attack probability before sigmoid)
    def forward(self, x):
        return self.fc(self.conv(x.transpose(1, 2))).squeeze(1)
        #x.transpose(1,2) swaps time and feature dimensions because Conv1d expects
        #(batch, channels, length) but our input is (batch, length, channels)
        #squeeze(1) removes the extra dimension so the output is a 1D vector of scores


class CNN_Attention(nn.Module):
    def __init__(self, feat_dim=12, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        #same conv block as CNN_Only to extract local features from the packet sequence
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        #attention layer on top of CNN so the model can also look at which packets in the
        #sequence matter most globally not just locally like the CNN does
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        #same classification head as CNN_Only

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        #run through conv then transpose back to (batch, time, channels) for attention
        return self.fc(self.multi_head_attn(h)).squeeze(1)
        #attention collapses the sequence into one vector then fc gives the final score


class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self, feat_dim=12, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),      nn.ReLU(), nn.BatchNorm1d(128),
        )
        #same CNN feature extractor as the other two models
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        #BiLSTM reads the packet sequence both forward and backward
        #forward pass learns "what came before this packet"
        #backward pass learns "what comes after this packet"
        #hidden size is 64 but since bidirectional doubles it the output is 128 per time step
        #batch_first=True means input shape is (batch, time, features) which is more natural
        self.multi_head_attn = MultiHeadAttention(128, num_heads)
        #attention on top of BiLSTM output so the model decides which time steps are most important
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        #same final classification head

    def forward(self, x):
        h, _ = self.bilstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        #first CNN extracts local features then BiLSTM processes the whole sequence in both directions
        #h contains the BiLSTM output for every time step and _ is the hidden state we don't need
        return self.fc(self.multi_head_attn(h)).squeeze(1)
        #attention picks the most important time steps then fc gives the final score
def build_model(model_type):
    if model_type == "cnn_only":
        return CNN_Only(feat_dim=FEAT_DIM, seq_len=SEQ_LEN)
    elif model_type == "cnn_attention":
        return CNN_Attention(feat_dim=FEAT_DIM, num_heads=4)
    elif model_type == "cnn_bilstm_attn":
        return CNN_BiLSTM_Attn(feat_dim=FEAT_DIM, num_heads=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
#this is a factory function that creates the right model object based on the name string
#it makes the code cleaner because we can just call build_model("cnn_only") instead of
#writing a big if-else every time we need to create a model

def is_tcp_mqtt_port_df(df: pd.DataFrame) -> pd.Series:
    proto = df.get("Protocol", pd.Series([""] * len(df))).astype(str).str.upper()
    info = df.get("Info", pd.Series([""] * len(df))).astype(str).str.upper()
    #we use .get() with a default so if the column is missing the code won't crash
    #we convert to uppercase so "tcp" and "TCP" and "Tcp" all match the same way
    mask_tcp_only = (proto == "TCP")
    #True for rows where Wireshark classified the packet as pure TCP (not MQTT)
    mask_mqtt_ports = info.str.contains(r"\b1883\b|\b8883\b", regex=True, na=False)
    #\b is a word boundary so we don't accidentally match things like "11883" or "18830"
    #na=False means rows with missing Info are treated as False (not kept)
    return mask_tcp_only & mask_mqtt_ports
    #both conditions must be true so only TCP packets talking on MQTT ports are kept
    #this is identical to merge_original_dataset.py so the real hardware data is filtered
    #in exactly the same way as the simulation training data — this is very important for
    #transfer learning because the model must see data in the same format it was trained on


def read_csv_features(csv_path: Path):
    print(f"[INFO] Reading CSV: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        #if the file can't be opened (corrupted, wrong path, wrong format) we return empty
        #instead of crashing the whole program
        return []

    df.columns = [c.strip() for c in df.columns]
    #strip whitespace from column names because sometimes CSV exports add spaces
    required_cols = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]

    if any(c not in df.columns for c in required_cols):
        print(f"[ERROR] Missing required columns in CSV: {required_cols}")
        print(f"[ERROR] Available columns: {list(df.columns)}")
        return []
    #if any required column is missing we can't build features so we stop early
    #this is the basic safety rule we always apply before processing any file

    total_rows = len(df)
    df = df.loc[is_tcp_mqtt_port_df(df), required_cols].copy()
    kept_rows = len(df)
    #apply the same TCP+MQTT port filter we use everywhere else
    #.copy() is important because it stops pandas from warning us about editing a slice

    print(f"[FILTER STATS]")
    print(f"  Total rows in csv                        : {total_rows:>10,}")
    print(f"  KEPT - Protocol=TCP and port 1883/8883  : {kept_rows:>10,}")

    if df.empty:
        return []
    #if nothing passed the filter then there is no useful data to return

    df["file_name"] = csv_path.name
    df = df.sort_values("Time").reset_index(drop=True)
    #sort by time so packets are in chronological order which is important for
    #building sequences that represent a real time window of network traffic
    #reset_index so row numbers are clean after sorting

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0)
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(0.0)
    df["Time"] = df["Time"] - df["Time"].min()
    #make Time relative (starting from 0) so it matches how we processed training data
    df["time_delta"] = df["Time"].diff().fillna(0.0)
    #time_delta is the gap between consecutive packets
    #this is an important feature because attack floods have very small time deltas
    #.diff() gives None for the first row so we fill that with 0

    info = df["Info"].astype(str).str.upper()
    ports = info.str.extract(r"(\d+)\s*→\s*(\d+)")
    sport = pd.to_numeric(ports[0], errors="coerce")
    dport = pd.to_numeric(ports[1], errors="coerce")
    #we use regex to extract the source port and destination port from the Info column
    #the arrow → is what Wireshark uses in Info strings like "44560 → 1883 [SYN]"
    #errors="coerce" turns any non-numeric extraction into NaN instead of crashing

    df["has_mqtt_port"] = ((sport.isin(MQTT_PORTS)) | (dport.isin(MQTT_PORTS))).fillna(False).astype(int)
    #1 if either the source or destination port is an MQTT port, 0 otherwise
    df["to_mqtt"] = (dport.isin(MQTT_PORTS)).fillna(False).astype(int)
    #1 if the packet is being sent TO the MQTT broker (client to broker direction)
    df["from_mqtt"] = (sport.isin(MQTT_PORTS)).fillna(False).astype(int)
    #1 if the packet is being sent FROM the MQTT broker (broker to client direction)
    df["flag_syn"] = info.str.contains(r"\bSYN\b", regex=True, na=False).astype(int)
    #SYN flag appears when a new TCP connection is being started
    #in SYN flooding attacks you see many SYN packets without corresponding ACKs
    df["flag_ack"] = info.str.contains(r"\bACK\b", regex=True, na=False).astype(int)
    #ACK flag appears in normal replies confirming received data
    df["flag_fin"] = info.str.contains(r"\bFIN\b", regex=True, na=False).astype(int)
    #FIN flag appears when a connection is being properly closed
    df["flag_rst"] = info.str.contains(r"\bRST\b", regex=True, na=False).astype(int)
    #RST flag appears when a connection is abruptly terminated (reset)
    #RST flooding is a type of attack where the attacker forcefully kills connections
    df["flag_psh"] = info.str.contains(r"\bPSH\b", regex=True, na=False).astype(int)
    #PSH flag means the sender wants the data sent immediately to the application
    df["flag_urg"] = info.str.contains(r"\bURG\b", regex=True, na=False).astype(int)
    #URG flag is for urgent data (rare in normal traffic so could indicate anomaly)

    feature_cols = [
        "Time", "time_delta", "Length",
        "has_mqtt_port",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
        "to_mqtt", "from_mqtt",
    ]
    #these are the exact 12 features we use so the order must match the order used in training
    #if the order changes the model will read the wrong feature in each position and break

    feats = df[feature_cols].to_numpy(dtype=np.float32)
    print(f"  Feature vectors extracted               : {len(feats):>10,}")
    return [row for row in feats]
    #return as a list of rows so each element is one packet's feature vector
    #this makes it easy to slide a window over them in build_sequences

def build_sequences(feats, seq_len=SEQ_LEN, step=STEP):
    seqs = []
    for i in range(0, len(feats) - seq_len + 1, step):
        seqs.append(np.stack(feats[i : i + seq_len], axis=0))
    print(f"[INFO] Built {len(seqs):,} sequences (all label=0 / normal)")
    return seqs
#this is the sliding window that groups packets into sequences
#we start at position 0 and take seq_len=20 consecutive packets then move forward by step=5
#so the first sequence is packets 0-19, the second is 5-24, the third is 10-29 and so on
#this creates overlapping sequences which is good because it means we don't miss patterns
#that happen at the boundary between two non-overlapping windows
#all sequences from this real hardware capture are labeled 0 (normal) because
#we captured this data from a healthy broker with no attacks happening

def refit_scaler(feats, scaler_out_path: Path):
    continuous = np.array([f[[0, 1, 2]] for f in feats], dtype=np.float32)
    new_scaler = StandardScaler()
    new_scaler.fit(continuous)
    with open(scaler_out_path, "wb") as f:
        pickle.dump(new_scaler, f)
    print(f"[OK] New scaler saved -> {scaler_out_path}")
    return new_scaler
#the scaler normalizes the three continuous features (Time, time_delta, Length)
#so they all have mean=0 and standard deviation=1
#we MUST refit the scaler on real hardware data because real hardware may have
#very different timing and packet sizes compared to the simulation dataset
#if we used the old scaler the normalization would be wrong and the model would get bad inputs
#we only scale the 3 continuous features (indices 0,1,2) because the flag features
#(indices 3-11) are already binary (0 or 1) so they don't need normalization
#pickle.dump saves the scaler to disk so the live deployment script can load and use it

def freeze_backbone(model, model_type: str):
    for param in model.conv.parameters():
        param.requires_grad = False
    if model_type == "cnn_bilstm_attn":
        for param in model.bilstm.parameters():
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable params: {trainable:,}  |  Frozen params: {frozen:,}")
#this function implements the core idea of transfer learning which is
#"freeze what the model already knows and only train what needs to adapt"
#the CNN layers (backbone) learned to detect general TCP patterns from the big simulation dataset
#we freeze them by setting requires_grad=False so their weights don't change during fine-tuning
#for cnn_bilstm_attn we also freeze the BiLSTM since it also learned sequence patterns already
#only the fully connected (fc) classifier layers remain trainable
#this prevents the model from forgetting what it learned (called catastrophic forgetting)
#and it also means we need much less data to fine-tune because fewer parameters need updating
#we print how many parameters are trainable vs frozen so we can verify the freeze worked
def scale_sequences(seqs, scaler):
    continuous_idx = [0, 1, 2]
    scaled = []
    for seq in seqs:
        s = seq.copy()
        s[:, 0] = s[:, 0] - s[:, 0].min()
        #make Time relative within each sequence so the model sees time from 0
        #this is the same preprocessing we do in all training scripts
        s[:, continuous_idx] = scaler.transform(s[:, continuous_idx])
        #apply StandardScaler to normalize Time, time_delta and Length
        scaled.append(s)
    return np.array(scaled, dtype=np.float32)
#we process each sequence one at a time because we need to make Time relative
#within each individual sequence (not globally across all sequences)
#then we apply the new real-hardware scaler to normalize the continuous features
#.copy() is important to avoid modifying the original sequence data

def chunked_normal_accuracy(model, X, chunk=EVAL_CHUNK):
    n_correct = 0
    n_total   = 0
    for start in range(0, len(X), chunk):
        xb = torch.tensor(X[start : start + chunk]).to(DEVICE)
        with torch.no_grad():
            preds = (torch.sigmoid(model(xb)) < 0.5).long()
        n_correct += int(preds.sum().item())
        n_total   += len(xb)
        del xb, preds
    return n_correct / max(n_total, 1)
#this function checks how often the model correctly predicts "normal" (label=0)
#on the real hardware sequences after each fine-tuning epoch
#we process in chunks of 512 instead of all at once to avoid running out of RAM
#torch.no_grad() means we don't store gradients during evaluation so it's faster and uses less memory
#torch.sigmoid turns the raw model output into a probability between 0 and 1
#if probability < 0.5 we predict normal (0) so we count it as correct since all hardware data is normal
#del xb, preds frees the GPU/CPU memory after each chunk so we don't run out
#we return accuracy as a fraction between 0 and 1 (so 0.95 means 95% correct)

def fine_tune_detector(model, seqs, scaler, epochs=FT_EPOCHS, lr=FT_LR, batch=FT_BATCH):
    print(f"  Scaling {len(seqs):,} sequences...")
    X         = scale_sequences(seqs, scaler)
    y         = np.zeros(len(X), dtype=np.float32)
    #all labels are 0 (normal) because this is real hardware normal traffic
    #we are teaching the model that this type of traffic should be classified as safe
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    #Adam optimizer only updates parameters where requires_grad=True
    #which means only the unfrozen fc layers will be updated
    #filter(lambda p: p.requires_grad, ...) selects only the trainable parameters
    criterion = nn.BCEWithLogitsLoss()
    #BCEWithLogitsLoss is binary cross-entropy loss combined with sigmoid
    #it measures how wrong the model's prediction is compared to the true label
    #lower loss means the model is doing better
    indices   = list(range(len(X)))

    for ep in range(1, epochs + 1):
        model.train()
        #model.train() enables dropout so it works during training
        random.shuffle(indices)
        #shuffle the order we show data each epoch so the model doesn't memorize the order
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(indices), batch):
            idx = indices[start : start + batch]
            xb  = torch.tensor(X[idx]).to(DEVICE)
            yb  = torch.tensor(y[idx]).to(DEVICE)
            optimizer.zero_grad()
            #clear the gradients from the last batch so they don't accumulate
            loss = criterion(model(xb), yb)
            #compute how wrong the model was on this batch
            loss.backward()
            #backpropagate: compute how much each parameter contributed to the error
            optimizer.step()
            #update the trainable parameters to reduce the error
            total_loss += loss.item()
            n_batches  += 1
            del xb, yb, loss
            #free memory after each batch

        model.eval()
        #model.eval() disables dropout so evaluation is deterministic
        acc = chunked_normal_accuracy(model, X)
        print(f"  Epoch {ep:02d}/{epochs} | loss={total_loss/max(n_batches,1):.4f} | normal_acc={acc*100:.1f}%")
        #we print both loss and accuracy after every epoch so we can see if the model is improving
        #normal_acc should go up toward 100% as the model learns that real hardware traffic is normal

    model.eval()
    return model
    
def main():
    print("=" * 65)
    print("  NeuroGuard Transfer Learning - Real Hardware Fine-Tuning")
    print("=" * 65)
    print(f"  Device  : {DEVICE}")
    print(f"  CSV     : {PCAP_FILE}")
    print(f"  Out dir : {OUT_DIR}")
    print()
    print("  Filter (identical to training):")
    print("    KEEP   Protocol=TCP  AND  port 1883/8883 in Info")
    print()

    if not PCAP_FILE.exists():
        print(f"[ERROR] CSV file not found: {PCAP_FILE}")
        return
    #check the real hardware CSV file exists before we do anything else
    #if it's missing we print a clear error and stop so the user knows exactly what to fix

    feats = read_csv_features(PCAP_FILE)
    #read the CSV and extract all 12 features for every filtered packet

    if len(feats) < SEQ_LEN:
        print(f"[ERROR] Not enough filtered CSV rows ({len(feats)}). Need at least {SEQ_LEN}.")
        return
    #we need at least 20 packets to build even one sequence so if there is less we can't proceed

    seqs = build_sequences(feats)
    #slide the 20-packet window across the feature list to build all sequences

    if len(seqs) < 10:
        print(f"[ERROR] Not enough sequences ({len(seqs)}). Capture more traffic.")
        return
    #10 sequences is the absolute minimum to run fine-tuning but in practice we want many more
    #if there are fewer than 10 sequences the model can't meaningfully update its weights

    print()

    for det_name, cfg in DETECTOR_CONFIGS.items():
        print(f"{'─'*65}")
        print(f"  Fine-tuning: {det_name}")
        print(f"{'─'*65}")
        #loop over all three detector types and fine-tune each one with the same real hardware data

        print("[STEP 1] Refitting scaler on real hardware traffic...")
        new_scaler = refit_scaler(feats, cfg["scaler_out"])
        #STEP 1: fit a fresh StandardScaler on the real hardware packet features
        #this is critical because the simulation dataset may have had very different
        #timing distributions and packet sizes compared to our actual Raspberry Pi setup

        print(f"[STEP 2] Loading pre-trained detector from {cfg['ckpt_in']}")
        model = build_model(cfg["type"]).to(DEVICE)
        model.load_state_dict(torch.load(cfg["ckpt_in"], map_location=DEVICE))
        print("         Pre-trained weights loaded.")
        #STEP 2: create a fresh model with the correct architecture then
        #load the weights that were trained on the simulation dataset
        #map_location=DEVICE makes sure the weights load correctly even if they were
        #saved on GPU but we are now running on CPU (or vice versa)

        print("[STEP 3] Freezing CNN backbone (only FC layers will update)...")
        freeze_backbone(model, cfg["type"])
        #STEP 3: freeze the CNN (and BiLSTM for cnn_bilstm_attn) layers
        #only the final classifier (fc) layers will be updated during fine-tuning
        #this is the key principle of transfer learning: keep the useful general features
        #and only adapt the decision boundary to the new domain (real hardware)

        print(f"[STEP 4] Fine-tuning for {FT_EPOCHS} epochs on {len(seqs):,} sequences...")
        model = fine_tune_detector(model, seqs, new_scaler)
        #STEP 4: run 10 epochs of gradient descent on the real hardware sequences
        #only the unfrozen fc layers will update so the model adapts its decision boundary
        #without forgetting the general TCP attack patterns it learned in training

        torch.save(model.state_dict(), cfg["ckpt_out"])
        print(f"[OK] Fine-tuned detector saved -> {cfg['ckpt_out']}")
        print()
        #save the fine-tuned model weights to a new file (ending in _ft.pt)
        #we keep the original model file untouched so we can always go back to it

    print("=" * 65)
    print("  Transfer learning DONE.")
    print()
    print("  Files saved:")
    for det_name, cfg in DETECTOR_CONFIGS.items():
        print(f"    {cfg['scaler_out'].name}")
        print(f"    {cfg['ckpt_out'].name}")
    print()
    print("  Next step: run  4b_rl_retrain_mixed.py")
    print("=" * 65)
    #print a summary of all the files that were created and what to run next


if __name__ == "__main__":
    main()
