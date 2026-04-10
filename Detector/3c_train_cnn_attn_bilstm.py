#cnn + bilstm + multi-head attention model
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import math
# importing the libraries we need for:
# reading the csv, building a deep learning model and training and evaluation metrics
# StandardScaler for feature normalization

OUT_DIR = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")
# the folder where we saved our dataset and split files and where we will save the model

TRAIN_IDX = OUT_DIR / "train_idx.npy"
VAL_IDX = OUT_DIR / "val_idx.npy"
TEST_IDX = OUT_DIR / "test_idx.npy"
# these files contain the row numbers for train, val & test
# they make sure we always use the same split (reproducible) and avoid leakage
CSV_ALL = OUT_DIR / "mqtt_packets_labeled.csv"
# the full labeled packets csv (raw data)
MODEL_OUT = OUT_DIR / "detector_cnn_bilstm_attn.pt"
SCALER_OUT = OUT_DIR / "scaler_cnn_bilstm_attn.pkl"
RESULTS_OUT = OUT_DIR / "results_cnn_bilstm_attn.txt"
# where the best trained model weights and scaler will be saved

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# this means use gpu if available because it trains faster otherwise use cpu
BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-3
# batch size means how many sequences per step
# epochs means how many full passes on the training set
# learning rate means how fast the model updates its weights
# sequence settings
SEQ_LEN = 20
STEP = 5
# we build each sample as a window of 20 packets
# step=5 means we slide the window forward by 5 packets each time to reduce overlap (there will be an overlap)
class SeqDataset(Dataset):
    def __init__(self, X, y):
        # here we convert the data into a format that PyTorch understands and uses it for training
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # labels must be floats because BCEWithLogitsLoss() requires floats
        # BCEWithLogitsLoss() measures how wrong the model is when predicting attack vs normal

    def __len__(self):
        # it says how many samples total which is the N sequences
        return self.X.shape[0]

    def __getitem__(self, i):
        # this returns one sequence and its label
        return self.X[i], self.y[i]


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


class CNN_BiLSTM_Attn(nn.Module):
    # full model: CNN for local patterns + BiLSTM for sequential behavior + Multi-Head Attention for focusing
    # this is the most powerful model combining all three mechanisms
    def __init__(self, feat_dim, num_heads=4):
        # feat_dim is the number of features per packet (12 in our case)
        # num_heads is the number of attention heads (default 4)
        super().__init__()
        # registering the model with PyTorch

        # CNN part: extracts local patterns from packets
        self.conv = nn.Sequential(
            # this creates the CNN part of our model
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            # this will look at small groups of packets to detect short-term patterns
            # feat_dim is the number of input features per packet (12 features)
            # 128 is the output channels, the number of filters
            # means the model learns 128 different local patterns
            # kernel_size=3 means how many packets are looked at at once
            # padding=1 means keeping the sequence length the same to prevent losing packets at the edges
            nn.ReLU(),
            # activation function that adds non-linearity so patterns can be complex
            nn.BatchNorm1d(128),
            # batch normalization for training stability
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            # this learns deeper and stronger local patterns
            # input now is 128 learned features and the output stays 128
            nn.ReLU(),
            nn.BatchNorm1d(128)
            # again this allows complex behavior learning
        )

        # BiLSTM part: captures sequential dependencies in both directions
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        # LSTM is a type of recurrent neural network that remembers patterns over time
        # 128 is the input size (from CNN output)
        # 64 is the hidden size (how much memory the LSTM has)
        # batch_first=True means data comes in as (batch, seq_len, features)
        # bidirectional=True means the LSTM reads the sequence forwards AND backwards
        # this helps it understand context from both past and future packets
        # output will be 64*2=128 because bidirectional doubles the hidden size

        # Multi-Head Attention part: focuses on the most important packets from multiple perspectives
        self.multi_head_attn = MultiHeadAttention(128, num_heads=num_heads)
        # 128 is the hidden dimension from BiLSTM output (64*2)
        # num_heads allows learning different attention patterns simultaneously
        # attention learns which packets matter most for the final decision

        # Final classification layers
        self.fc = nn.Sequential(
            # this creates the final part of the model that makes the decision
            nn.Linear(128, 128),
            # compress the attention-pooled features
            nn.ReLU(),
            # activation for non-linearity
            nn.Dropout(0.3),
            # dropout to prevent overfitting
            # randomly turns off 30% of neurons during training
            # this forces the model to learn robust patterns instead of memorizing
            nn.Linear(128, 1)
            # final output: one score for attack vs normal
        )

    def forward(self, x):
        # this function defines how data flows through the model
        # x arrives as (batch, timesteps, features)

        # Step 1: CNN processes local patterns
        x = x.transpose(1, 2)
        # reorder data so CNN can process it
        # changes shape from (B, T, F) to (B, F, T)
        # because CNN expects features first and time last
        x = self.conv(x)
        # CNN finds short term packet patterns
        # output shape: (B, 128, T)

        # Step 2: BiLSTM processes sequential patterns
        x = x.transpose(1, 2)
        # change back to (B, T, 128) for LSTM
        # LSTM expects (batch, seq_len, features)
        x, _ = self.bilstm(x)
        # BiLSTM learns how packets relate to each other over time
        # reads sequence forwards and backwards to understand full context
        # output shape: (B, T, 128) where 128 = 64*2 from bidirectional

        # Step 3: Multi-Head Attention focuses on important packets from multiple perspectives
        x = self.multi_head_attn(x)
        # attention decides which packets are most important
        # creates a weighted summary of the entire sequence
        # output shape: (B, 128)

        # Step 4: Final classification
        return self.fc(x).squeeze(1)
        # make final decision: attack or normal
        # squeeze(1) removes extra dimension so output shape is correct


# now this part is about the evaluation metrics
def metrics(y_true, y_prob, thr=0.5):
    # convert probabilities into 0/1 predictions using threshold
    # so this function calculates how good the model is
    # y_true is the real labels (attack or normal)
    # y_prob is the model's predicted probabilities
    # thr=0.5 is the decision threshold
    y_pred = (y_prob >= thr).astype(int)
    # now this converts probabilities into final decisions
    # so if probability ≥ 0.5 then it's an attack (1)
    # if probability < 0.5 then normal (0)
    # now we will compute common classification metrics
    return {
        "acc": accuracy_score(y_true, y_pred),
        # this measures how many predictions are correct overall
        "prec": precision_score(y_true, y_pred, zero_division=0),
        # measures how many predicted attacks are actually attacks
        "rec": recall_score(y_true, y_pred, zero_division=0),
        # this measures how many real attacks were correctly detected
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # this balances the precision and recall into one score
        "cm": confusion_matrix(y_true, y_pred),
        # this shows counts of true normal, false attack, missed attack and correct attack
    }


def add_features(df: pd.DataFrame):
    # this creates the feature columns used by the model
    # we rebuild them after splitting so that preprocessing stays consistent
    feat = df.copy()

    feat["Time"] = pd.to_numeric(feat["Time"], errors="coerce").fillna(0.0)
    feat["Length"] = pd.to_numeric(feat["Length"], errors="coerce").fillna(0.0)

    # normalize Time per file so model doesn't depend on absolute capture start
    feat["Time"] = feat.groupby("file_name")["Time"].transform(lambda s: s - s.min())

    feat["time_delta"] = feat.groupby("file_name")["Time"].diff().fillna(0.0)

    info = feat["Info"].astype(str).str.upper()

    # extract source and destination ports from pattern:
    # 1883 → 53415
    # 51781 → 1883
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

    # final 12 features
    feature_cols = [
        "Time", "time_delta", "Length",
        "has_mqtt_port",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
        "to_mqtt", "from_mqtt",
    ]

    return feat, feature_cols



def scale_features(X, scaler=None, fit=False):
    # normalize continuous features to have mean=0 and std=1
    # this improves training stability and convergence speed
    # X shape: (num_sequences, seq_len, num_features)

    # indices of continuous features that need scaling
    continuous_idx = [0, 1, 2]  # Time, time_delta, Length
    # binary features (has_1883 and flags) don't need scaling

    original_shape = X.shape
    # save original shape to restore after scaling

    # reshape to (num_sequences * seq_len, num_features) for scaling
    X_reshaped = X.reshape(-1, X.shape[-1])

    if fit:
        # fit scaler on training data only
        scaler = StandardScaler()
        X_reshaped[:, continuous_idx] = scaler.fit_transform(X_reshaped[:, continuous_idx])
        print(f"[INFO] Fitted scaler on continuous features: {continuous_idx}")
    else:
        # transform using existing scaler for val/test
        X_reshaped[:, continuous_idx] = scaler.transform(X_reshaped[:, continuous_idx])

    # reshape back to original shape
    X_scaled = X_reshaped.reshape(original_shape)

    return X_scaled, scaler


# now this function aims to prevent mixing packets from different capture files
# which would cause wrong learning
def make_sequences_per_file(df_feat: pd.DataFrame, feature_cols, seq_len=20, step=5):
    X_list = []
    y_list = []
    # we will store sequences and their labels here
    # build sequences inside each file_name separately
    # so we do not mix packets from different captures to prevent leakage
    for fname, g in df_feat.groupby("file_name", sort=False):
        g = g.sort_values("Time")
        # keep correct time order inside each file
        # packets are not mixed across files
        # time order is preserved
        X = g[feature_cols].to_numpy(dtype=np.float32)
        # take the feature columns and convert to float32 array
        # because neural networks require numeric tensor inputs for efficient computation and training
        y = g["label"].astype(int).to_numpy()
        # extracting the labels for that file as integers
        # because labels represent classes and not measurements

        if len(X) < seq_len:
            continue
        # if a file has fewer than seq_len rows skip it cuz we can't form a full sequence.
        # we should set a base rule that's why
        for start in range(0, len(X) - seq_len + 1, step):
            end = start + seq_len
            X_list.append(X[start:end])
            # adding the seq_len row chunk total shape is seq_len × 12
            y_list.append(int(y[start:end].max()))
            # sequence label is the max label in those seq_len rows
            # start is the index of the first packet in a sequence
            # start goes from 0 up to the last valid window
            # step controls overlap
            # overlap helps the model see how traffic changes
            # over time instead of seeing isolated chunks
            # label the sequence as attack if any packet inside the window is attack
            # this matches window-based detection

    if len(X_list) == 0:
        raise RuntimeError("No sequences were created. Try smaller SEQ_LEN or check your data.")
    # if nothing was created then sequence length may be too large or data is missing
    # so it's a safety check like the base rule
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.float32)
    # X becomes (num_sequences, seq_len, 12)
    # y becomes (num_sequences,)


# this function loads data safely using proper splitting and preprocessing
def load_group_split_sequences():
    print("[INFO] Loading full CSV...")
    full = pd.read_csv(CSV_ALL)
    # load the entire dataset that we made

    train_idx = np.load(TRAIN_IDX)
    val_idx = np.load(VAL_IDX)
    test_idx = np.load(TEST_IDX)
    # load the saved row numbers for train/val/test

    train_df = full.iloc[train_idx].copy()
    val_df = full.iloc[val_idx].copy()
    test_df = full.iloc[test_idx].copy()
    # slice the dataset into 3 parts using the saved row numbers

    train_feat, feature_cols = add_features(train_df)
    val_feat, _ = add_features(val_df)
    test_feat, _ = add_features(test_df)
    # now we'll build features for each split

    Xtr, ytr = make_sequences_per_file(train_feat, feature_cols, SEQ_LEN, STEP)
    Xva, yva = make_sequences_per_file(val_feat, feature_cols, SEQ_LEN, STEP)
    Xte, yte = make_sequences_per_file(test_feat, feature_cols, SEQ_LEN, STEP)
    # now build sequences per file for each split to prevent mixing files

    # apply feature scaling
    print("[INFO] Applying feature scaling...")
    Xtr, scaler = scale_features(Xtr, fit=True)
    # fit scaler on training data only to prevent data leakage
    Xva, _ = scale_features(Xva, scaler=scaler, fit=False)
    Xte, _ = scale_features(Xte, scaler=scaler, fit=False)
    # apply the same scaling to val and test using the training scaler

    print(f"[INFO] Sequences: train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,}")
    print(f"[INFO] Feature dim: {Xtr.shape[-1]}")
    return Xtr, ytr, Xva, yva, Xte, yte, scaler


# now the main function to train and test
def main():
    Xtr, ytr, Xva, yva, Xte, yte, scaler = load_group_split_sequences()
    # first loading prepared sequences for training/validation/testing
    feat_dim = Xtr.shape[-1]
    # this to make sure that the feature dimension must match model input

    # save the scaler for later use in deployment
    with open(SCALER_OUT, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved scaler to: {SCALER_OUT}")

    train_ds = SeqDataset(Xtr, ytr)
    val_ds = SeqDataset(Xva, yva)
    test_ds = SeqDataset(Xte, yte)
    # the first one creates the training dataset
    # Xtr is the training sequences the packet sequences
    # ytr is the training labels normal / attack
    # SeqDataset pairs each sequence with its label

    # now lets come to the imbalance handling
    # we use WeightedRandomSampler so the model does not ignore the minority class
    ytr_np = ytr.astype(int)
    class_counts = np.bincount(ytr_np)
    w0 = 1.0 / (class_counts[0] + 1e-9)
    w1 = 1.0 / (class_counts[1] + 1e-9)
    sample_weights = np.where(ytr_np == 0, w0, w1).astype(np.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    # this gives more chance to sample the rare class each epoch

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    # training batches of batch size (how many sequences are sent to the model at once)
    # train_ds is the training dataset which contains (sequence, label) pairs
    # sampler=sampler controls which samples(seq) are picked and in what order
    # it's often used for class imbalance and when sampler is used no shuffle is needed
    # num_workers=0 means data loading happens in the main process
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # no shuffle for val/test cuz the evaluation should be stable/repeatable
    # training needs randomness to learn better
    # but validation/testing need stability to measure fairly

    model = CNN_BiLSTM_Attn(feat_dim, num_heads=4).to(DEVICE)
    # build the model with 4 attention heads then move it to cpu/gpu (as our base rule)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # the optimizer is the part that teaches the model how to change its weights so it
    # makes fewer mistakes
    # lr is the learning rate which controls how big each update step is
    loss_fn = nn.BCEWithLogitsLoss()
    # we use BCEWithLogitsLoss cuz this loss function does two things at once:
    # it applies sigmoid internally
    # (the sigmoid maps model outputs to a 0–1 range
    # allowing them to be interpreted as probabilities for binary classification)
    # and it computes binary cross-entropy loss which measures how wrong the model is when predicting attack
    # vs normal
    # my model gives me a raw score not a probability so the BCEWithLogitsLoss is used
    # because it converts logits to probabilities internally and computes the correct binary loss

    best_f1 = -1.0
    # track best validation f1 score so we save the best model

    for epoch in range(1, EPOCHS + 1):
        # start the training phase with an epoch set training mode
        model.train()
        # tells the network it is learning so training specific behaviors like dropout are enabled
        # to stop the model from memorizing the data
        losses = []
        train_probs = []
        train_ys = []
        # collecting training outputs to compute train accuracy

        for xb, yb in train_loader:
            # this means go through the training data batch by batch
            # where
            # xb=a batch of input sequences (BATCH_SIZE sequences)
            # yb=the correct labels for that batch
            # now move batch to gpu/cpu
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            # we reset old errors, get the model's prediction and calculate how wrong it is for this batch
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            # backward() finds the mistakes and step() update model weights which means updates the model so it
            # makes fewer mistakes next time
            losses.append(loss.item())
            # this saves the loss value (how wrong the model was) for this batch
            train_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_ys.append(yb.cpu().numpy())
            # collecting train predictions and labels to compute train accuracy

        train_probs = np.concatenate(train_probs)
        train_ys = np.concatenate(train_ys)
        train_acc = accuracy_score(train_ys, (train_probs >= 0.5).astype(int))
        # compute training accuracy for this epoch

        # validate
        model.eval()
        # switching to evaluation mode
        with torch.no_grad():
            # this means do not learn just predict
            probs = []
            ys = []
            val_losses = []
            # collecting outputs to compute metrics
            for xb, yb in val_loader:
                # going through validation data batch by batch
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                # model makes predictions which are logits
                val_losses.append(loss_fn(logits, yb).item())
                # compute validation loss for this batch
                p = torch.sigmoid(logits).cpu().numpy()
                # convert logits to probabilities
                probs.append(p)
                # it means that save model predictions
                # then store for later evaluation
                ys.append(yb.cpu().numpy())

            probs = np.concatenate(probs)
            # combine batches into one big array
            ys = np.concatenate(ys)

        m = metrics(ys, probs)
        # compute validation metrics
        print(
            f"Epoch {epoch} | loss {np.mean(losses):.4f} | train acc {train_acc:.4f} | val acc {m['acc']:.4f} | val loss {np.mean(val_losses):.4f} | f1 {m['f1']:.4f} rec {m['rec']:.4f}")

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  [OK] saved best model: {MODEL_OUT}")
            # save best model so we keep the best generalizing version

    # testing
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    model.eval()
    # load the best saved model then evaluate it on test
    # then the test loop is the same as what we did in the validation:
    # we collect logits
    # then from sigmoid to probs
    # threshold to preds
    # compute accuracy/precision/recall/F1

    with torch.no_grad():
        probs = []
        ys = []
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
            ys.append(yb.numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)

    m = metrics(ys, probs)
    # compute final test metrics
    print("\n=== CNN + BiLSTM + MULTI-HEAD ATTENTION - TEST METRICS ===")
    print(f"Accuracy:  {m['acc']:.4f}")
    print(f"Precision: {m['prec']:.4f}")
    print(f"Recall:    {m['rec']:.4f}")
    print(f"F1 Score:  {m['f1']:.4f}")
    print("Confusion matrix:\n", m["cm"])

    # save results to file
    with open(RESULTS_OUT, "w") as f:
        f.write("=== CNN + BiLSTM + MULTI-HEAD ATTENTION - TEST METRICS ===\n")
        f.write(f"Accuracy:  {m['acc']:.4f}\n")
        f.write(f"Precision: {m['prec']:.4f}\n")
        f.write(f"Recall:    {m['rec']:.4f}\n")
        f.write(f"F1 Score:  {m['f1']:.4f}\n")
        f.write(f"Confusion matrix:\n{m['cm']}\n")
    print(f"\n[OK] Results saved to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()
    # run training when executing this file directly