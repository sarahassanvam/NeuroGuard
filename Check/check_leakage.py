#4 run checking only
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")
CSV_ALL = OUT_DIR / "mqtt_packets_labeled.csv"
TRAIN_IDX = OUT_DIR / "train_idx.npy"
VAL_IDX   = OUT_DIR / "val_idx.npy"
TEST_IDX  = OUT_DIR / "test_idx.npy"

print("[INFO] Loading CSV...")
df = pd.read_csv(CSV_ALL, usecols=["file_name", "label"])
#added label so we can also check class distribution per split

print("[INFO] Loading split indices...")
train_idx = np.load(TRAIN_IDX)
val_idx   = np.load(VAL_IDX)
test_idx  = np.load(TEST_IDX)

# Extract file names for each split
train_files = set(df.iloc[train_idx]["file_name"].unique())
val_files   = set(df.iloc[val_idx]["file_name"].unique())
test_files  = set(df.iloc[test_idx]["file_name"].unique())

print("\n    SPLIT STATISTICS    ")
print(f"Total rows in dataset: {len(df):,}")
print(f"Train rows: {len(train_idx):,} ({len(train_idx)/len(df)*100:.1f}%)")
print(f"Val rows:   {len(val_idx):,} ({len(val_idx)/len(df)*100:.1f}%)")
print(f"Test rows:  {len(test_idx):,} ({len(test_idx)/len(df)*100:.1f}%)")

print(f"\nTrain files: {len(train_files)}")
print(f"Val files:   {len(val_files)}")
print(f"Test files:  {len(test_files)}")

print("\n    FILE OVERLAP CHECK    ")
overlap_train_val = len(train_files & val_files)
overlap_train_test = len(train_files & test_files)
overlap_val_test = len(val_files & test_files)

print(f"Train ∩ Val:  {overlap_train_val} files")
print(f"Train ∩ Test: {overlap_train_test} files")
print(f"Val ∩ Test:   {overlap_val_test} files")

if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
    print("\n[OK] No file_name overlap the group split is leakage-safe")
else:
    print("\n[WARN] Overlap exists the prepare_split.py must be fixed")
    if overlap_train_val > 0:
        print(f"  → {overlap_train_val} files shared between train and val")
    if overlap_train_test > 0:
        print(f"  → {overlap_train_test} files shared between train and test")
    if overlap_val_test > 0:
        print(f"  → {overlap_val_test} files shared between val and test")

#and now an additional check to verify all indices are unique and within bounds
print("\n     INDEX INTEGRITY CHECK     ")
all_indices = np.concatenate([train_idx, val_idx, test_idx])
if len(all_indices) == len(set(all_indices)):
    print("All indices are unique so no row appears in multiple splits")
else:
    print("WARNING some rows appear in multiple splits")

if all_indices.max() < len(df) and all_indices.min() >= 0:
    print("All indices are within valid range")
else:
    print("WARNING some indices are out of bounds")
#now we will check class balance
print("\n    CLASS DISTRIBUTION    ")
labels = df["label"].values
print(f"Train attack%: {labels[train_idx].mean()*100:.2f}%")
print(f"Val attack%:   {labels[val_idx].mean()*100:.2f}%")
print(f"Test attack%:  {labels[test_idx].mean()*100:.2f}%")
print(f"Overall attack%: {labels.mean()*100:.2f}%")

print("\n[DONE] verification complete!")