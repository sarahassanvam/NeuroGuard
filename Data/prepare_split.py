import os
from pathlib import Path
#these two above are tools to work with file paths
#it's safer and cleaner path handling than hard-coding strings
import numpy as np
import pandas as pd
#numpy for arrays and pandas for csv handling these are the standard tools for ml data processing
from sklearn.model_selection import GroupShuffleSplit
#a splitter that respects groups we use it because it prevents data leakage
#by keeping related samples togetherr
OUT_DIR = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")
CSV_PATH = OUT_DIR / "mqtt_packets_labeled.csv"
#define dataset location

TRAIN_OUT = OUT_DIR / "train_idx.npy"
VAL_OUT   = OUT_DIR / "val_idx.npy"
TEST_OUT  = OUT_DIR / "test_idx.npy"
#the dir where we will save each split
print("[INFO] Loading CSV...")
df = pd.read_csv(CSV_PATH, usecols=["label", "file_name"])
#load only label and file_name cuz it's faster & less memory

#safety checks like the basic rule
if df.empty:
    raise RuntimeError("CSV is empty! Check if the first script ran successfully.")
if "file_name" not in df.columns or "label" not in df.columns:
    raise RuntimeError("Required columns missing! Check CSV format.")

unique_files = df["file_name"].nunique()
print(f"[INFO] Found {unique_files} unique files (groups)")
if unique_files < 10:
    print("WARNING: Very few unique files so splits may be imbalanced!")

y = df["label"].astype(int).values
#we will extract labels as int cuz ml models expect numeric labels
groups = df["file_name"].astype(str).values
#now we'll extract group ids cuz packets from the same file must stay in the same split to avoid data leakage
idx = np.arange(len(df))
#we'll create an index array cuz we will split row numbers not the actual csv what
#do i mean by that is that the CSV stays unchanged so we only just say:
#rows 0,2,5 go to training & rows 3,7 go to testing
#10% test
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
#this will create a rule for how the data will be split
#and this function will let all rows with the same file_name stay together
#n_splits it means that how many different random splits we want to generate
#test_size is the percentage of data used for testing
#and the last parameter we choose 42 so we can have the same split every time we run the code
trainval_idx, test_idx = next(gss1.split(idx, y=y, groups=groups))
#trainval_idx means rows allowed for training/validation
#test_idx means rows allowed ONLY for testing
#10% val so the split will be 80/10/10 so the 80 is for training
y_trainval = y[trainval_idx]
groups_trainval = groups[trainval_idx]
#we are now restricting labels and groups to remaining data because validation must come only from
#train+val and not test
#now we have 90% total of val and train so we will split them so 80% train and 10% val
#so we wrote 0.1111111 because 0.1111111 × 90% ≈ 10% so our final ratio is 80/10/10
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1111111, random_state=42)
train_idx, val_idx = next(gss2.split(trainval_idx, y=y_trainval, groups=groups_trainval))
#now this let's us choose which row positions go to train and which go to validation while keeping files together
#which means within the same file or group
#split the data into train and validation using group rules which is
#all rows with the same group value (same file_name) must stay together and it came from
#groups=groups_trainval
#and trainval_idx specify the row numbers we are allowed to split
#y=y_trainval labels of those rows
train_idx = trainval_idx[train_idx]
#now we will convert train positions into real row numbers from our original data
val_idx   = trainval_idx[val_idx]
#and again we will convert validation positions into real row numbers from the original data

np.save(TRAIN_OUT, train_idx)
np.save(VAL_OUT, val_idx)
np.save(TEST_OUT, test_idx)
#we will show the split sizes
print("[OK] Saved the row numbers for each split:")
print(" train:", len(train_idx))
print(" val  :", len(val_idx))
print(" test :", len(test_idx))

print("\n[CHECK] Label ratio:")
print(" train attack%:", y[train_idx].mean())
print(" val   attack%:", y[val_idx].mean())
print(" test  attack%:", y[test_idx].mean())

#and then we will verify that there is no file leakage
print("\n[CHECK] Verifying no file leakage between splits...")
train_files = set(groups[train_idx])
val_files = set(groups[val_idx])
test_files = set(groups[test_idx])

assert len(train_files & val_files) == 0, "ERROR: Files overlap between train and val!"
assert len(train_files & test_files) == 0, "ERROR: Files overlap between train and test!"
assert len(val_files & test_files) == 0, "ERROR: Files overlap between val and test!"
print(" No file leakage detected so splits are clean!")

print("\n[DONE]")