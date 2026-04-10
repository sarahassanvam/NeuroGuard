#1 run tcp with MQTT ports (1883/8883)
import os
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

BASE_DIR = r"C:\Users\User\Downloads\archive (1)\DoS-DDoS-MQTT-IoT_Dataset"
OUT_DIR = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
#ensure output folder exists (so saving files won't fail)
OUT_CSV = OUT_DIR / "mqtt_packets_labeled.csv"
FEATURE_LIST_TXT = OUT_DIR / "feature_columns.txt"

ATTACK_FOLDERS = [
    "SYN TCP Flooding",
    "Invalid Subscription Flooding",
    "Delayed Connect Flooding",
    "Connect Flooding with WILL payload",
    "Basic Connect Flooding",
]

MAX_ROWS_PER_FILE = 40_000
#not reading too many rows from one file to avoid huge ram usage


def find_ddos_csv_clean_dir(attack_folder: str) -> str:
    ddos_root = os.path.join(BASE_DIR, attack_folder, "DDoS")
    csv_clean_dir = os.path.join(ddos_root, "csv_clean")
    #we will work only on DDoS/csv_clean folders(i created them after i ternid them to csv
    if os.path.isdir(csv_clean_dir):
        return csv_clean_dir
    #if the folder exists return it
    else:
        #if missing try without csv_clean subfolder(just cuz we should have a base rule)
        print(f"[WARNING] Missing csv_clean subfolder for: {attack_folder}")
        print(f"          Tried: {csv_clean_dir}")
        print(f"          Trying parent DDoS folder instead...")
        if os.path.isdir(ddos_root):
            return ddos_root
        #if DDoS folder exists use it directly
        else:
            #if even DDoS folder is missing raise error
            raise FileNotFoundError(
                f"Missing DDoS folder for: {attack_folder}\nLooked for: {ddos_root}"
            )


def is_tcp_mqtt_port_df(df: pd.DataFrame) -> pd.Series:
    proto = df.get("Protocol", pd.Series([""] * len(df))).astype(str).str.upper()
    info = df.get("Info", pd.Series([""] * len(df))).astype(str).str.upper()
    #only TCP rows
    mask_tcp_only = (proto == "TCP")
    #only rows that show MQTT ports in info (1883 / 8883)
    mask_mqtt_ports = info.str.contains(r"\b1883\b|\b8883\b", regex=True, na=False)
    #keep only if both conditions are true (tcp and mqtt port)
    return mask_tcp_only & mask_mqtt_ports


def load_and_filter_csv(file_path: str, label: int, attack_type: str, max_rows_per_file: int | None):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        return pd.DataFrame(), 0, 0, 0
    #if file can't be read so return empty (skip it)
    #also return 0 for original and kept counts

    original_count = len(df)
    #store the original row count before filtering

    df.columns = [c.strip() for c in df.columns]
    #clean column names

    required_cols = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
    #ensuring required columns exist
    if any(c not in df.columns for c in required_cols):
        print(f"[WARNING] Missing required columns in {file_path}")
        print(f"          Available columns: {list(df.columns)}")
        return pd.DataFrame(), original_count, 0, 0
    #if any are missing then stop processing the file (skip it)

    df = df.loc[is_tcp_mqtt_port_df(df), required_cols].copy()
    #we will work on a copy so we don't modify the original dataframe
    #keep only tcp rows with mqtt port and only the needed columns

    if df.empty:
        return df, original_count, 0, 0
    #if no tcp rows with mqtt port exist then return empty
    tcp_mqttport_count = len(df)
    #count how many tcp rows with mqtt port we found before sampling

    if max_rows_per_file is not None and len(df) > max_rows_per_file:
        df = df.sample(n=max_rows_per_file, random_state=42)
    #we will stop building a huge dataset from one file to also avoid huge ram usage
    #seed idea is using random_state means reproducible sampling

    kept_count = len(df)
    #final count after sampling

    df["label"] = int(label)
    #label when 0 normal and 1 attack

    df["attack_type"] = attack_type
    df["file_name"] = os.path.basename(file_path)
    #keep file_name as metadata so later we can split by file_name to ensure no leakage

    return df, original_count, tcp_mqttport_count, kept_count


def save_feature_list():
    #tcp 12 features
    #these are tcp flag features we expect to extract from the info column
    feature_cols = [
        "Time", "time_delta", "Length",
        "has_mqtt_port",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
        "to_mqtt", "from_mqtt",
    ]
    #first 3 features which are Time, time_delta and Length
    #has_mqtt_port replaces
    #plus 6 tcp flags and 2 direction flags in total we will have 12 features total

    with open(FEATURE_LIST_TXT, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")
    #save the list of feature column names
    print("[OK] Saved feature list:", FEATURE_LIST_TXT)


def build_tcp_mqttport_packet_dataset():
    all_parts = []
    #empty list to store all filtered parts before concatenation

    total_original = 0
    total_tcp_mqttport_before_sampling = 0
    total_kept = 0
    normal_count = 0
    normal_tcp_mqttport_before_sampling = 0
    attack_count = 0
    attack_tcp_mqttport_before_sampling = 0
    attack_type_tcp_mqttport_before_sampling = {}
    #counters for tracking row statistics

    #the normal
    normal_dir = os.path.join(BASE_DIR, "NormalData", "CSV")
    normal_files = glob.glob(os.path.join(normal_dir, "*.csv"))
    print(f"[INFO] Found {len(normal_files)} normal CSV files in: {normal_dir}")

    for f in tqdm(normal_files, desc="Normal TCP (MQTT port)"):
        part, orig, tcp_before, kept = load_and_filter_csv(
            f, label=0, attack_type="Normal", max_rows_per_file=MAX_ROWS_PER_FILE
        )
        #label 0 = normal
        total_original += orig
        total_tcp_mqttport_before_sampling += tcp_before
        normal_tcp_mqttport_before_sampling += tcp_before
        total_kept += kept
        if not part.empty:
            normal_count += len(part)
            all_parts.append(part)

    print(f"[INFO] Normal data collected: {normal_count:,} rows")

    #the attacks
    for attack_folder in ATTACK_FOLDERS:
        try:
            ddos_dir = find_ddos_csv_clean_dir(attack_folder)
            ddos_files = glob.glob(os.path.join(ddos_dir, "*.csv"))
            print(f"[INFO] {attack_folder}: {len(ddos_files)} CSV files in: {ddos_dir}")

            if len(ddos_files) == 0:
                print(f"[WARNING] No CSV files found for {attack_folder}")
                continue

            attack_rows_this_type = 0
            attack_tcp_before_this_type = 0

            for f in tqdm(ddos_files, desc=f"DDoS TCP (MQTT port) - {attack_folder}", leave=False):
                part, orig, tcp_before, kept = load_and_filter_csv(
                    f, label=1, attack_type=attack_folder, max_rows_per_file=MAX_ROWS_PER_FILE
                )
                #label 1 is the attack label
                total_original += orig
                total_tcp_mqttport_before_sampling += tcp_before
                attack_tcp_mqttport_before_sampling += tcp_before
                attack_tcp_before_this_type += tcp_before
                total_kept += kept

                if not part.empty:
                    attack_rows_this_type += len(part)
                    all_parts.append(part)

            attack_type_tcp_mqttport_before_sampling[attack_folder] = attack_tcp_before_this_type
            print(f"[INFO] {attack_folder}: collected {attack_rows_this_type:,} attack rows")
            attack_count += attack_rows_this_type

        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            continue

    if not all_parts:
        raise RuntimeError("No TCP rows with MQTT port found. Check BASE_DIR and CSV format.")
    #safety check like the base rule

    full_df = pd.concat(all_parts, ignore_index=True)
    #combine all parts into one dataset

    full_df.to_csv(OUT_CSV, index=False)
    #save the combined datasett

    print(f"\n{'=' * 60}")
    print(f"[OK] Saved: {OUT_CSV}")
    print(f"{'=' * 60}")
    print(f"[STATS] Total rows in original CSVs: {total_original:,}")
    print(f"[STATS] Total TCP rows with MQTT port before sampling: {total_tcp_mqttport_before_sampling:,}")
    print(f"[STATS]   - Normal TCP (MQTT port) rows before sampling: {normal_tcp_mqttport_before_sampling:,}")
    print(f"[STATS]   - Attack TCP (MQTT port) rows before sampling: {attack_tcp_mqttport_before_sampling:,}")
    for attack_folder in ATTACK_FOLDERS:
        if attack_folder in attack_type_tcp_mqttport_before_sampling:
            print(f"[STATS]     * {attack_folder}: {attack_type_tcp_mqttport_before_sampling[attack_folder]:,}")
    print(f"[STATS] Total rows kept (after filtering & sampling): {len(full_df):,}")
    if total_original > 0:
        print(f"[STATS] Percentage kept: {100 * len(full_df) / total_original:.2f}%")
    print(f"[STATS] Normal rows: {normal_count:,}")
    print(f"[STATS] Attack rows: {attack_count:,}")
    print(f"{'=' * 60}")
    print("\n[LABEL DISTRIBUTION]")
    print(full_df["label"].value_counts())
    print(f"{'=' * 60}")
    #show detailed statistics about what was kept

    save_feature_list()
    #save feature list file for the next stage


if __name__ == "__main__":
    build_tcp_mqttport_packet_dataset()
    print("[DONE] Data preparation finished.")