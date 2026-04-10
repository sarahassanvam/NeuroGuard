#6 run
import numpy as np
import torch
from pathlib import Path
from collections import Counter
import time  #i added for response time measurement

from rl_env import (
    NeuroGuardRLEnv,
    build_rl_data_from_csv,
    ACTION_NAMES,
    A_ALLOW,
    A_ALERT_ONLY,
    A_ESCALATE,
    A_DEESCALATE,
)

OUT_DIR = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")

CSV_ALL = OUT_DIR / "mqtt_packets_labeled.csv"
TRAIN_IDX = OUT_DIR / "train_idx.npy"
VAL_IDX = OUT_DIR / "val_idx.npy"
TEST_IDX = OUT_DIR / "test_idx.npy"

#this dictionary maps detector names to their checkpoint files
DETECTOR_CONFIGS = {
    "cnn_only": {
        "ckpt": OUT_DIR / "detector_cnn_only.pt",
        "scaler": OUT_DIR / "scaler_cnn_only.pkl",
        "type": "cnn_only",
        "rl_model": OUT_DIR / "rl_policy_best_ACCEPTED_cnn_only.pt"
    },
    "cnn_attn": {
        "ckpt": OUT_DIR / "detector_cnn_attention.pt",
        "scaler": OUT_DIR / "scaler_cnn_attention.pkl",
        "type": "cnn_attention",
        "rl_model": OUT_DIR / "rl_policy_best_ACCEPTED_cnn_attn.pt"
    },
    "cnn_bilstm_attn": {
        "ckpt": OUT_DIR / "detector_cnn_bilstm_attn.pt",
        "scaler": OUT_DIR / "scaler_cnn_bilstm_attn.pkl",
        "type": "cnn_bilstm_attn",
        "rl_model": OUT_DIR / "rl_policy_best_ACCEPTED_cnn_bilstm_attn.pt"
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 20
STEP = 5
MAX_STEPS_EP = 200

STATE_DIM = 16
N_ACTIONS = 16
# these thresholds define how safe a model must be to be accepted
MAX_FPR = 0.05
MAX_FNR = 0.05

# actions that should not be counted as real mitigation
NON_MITIGATE_ACTIONS = {
    A_ALLOW,
    A_ALERT_ONLY,
    A_ESCALATE,
    A_DEESCALATE,
}


class QNet(torch.nn.Module):  # from the train
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# now this function evaluates ONE detector
def evaluate_single_detector(detector_name, detector_config, data):
    print(f"\n{'=' * 80}")
    print(f"[INFO] Evaluating detector: {detector_name}")
    print(f"{'=' * 80}\n")

    # get the detector checkpoint and scaler paths
    DETECTOR_CKPT = detector_config["ckpt"]
    SCALER_CKPT = detector_config["scaler"]
    DETECTOR_TYPE = detector_config["type"]
    RL_MODEL = detector_config["rl_model"]

    # check if RL model exists
    if not RL_MODEL.exists():
        print(f"[WARNING] RL model not found: {RL_MODEL}")
        print(f"Skipping {detector_name}...")
        return None

    Xte, yte, fte = (
        data["test"]["X"],
        data["test"]["y"],
        data["test"]["files"],
    )
    # the same as before
    env = NeuroGuardRLEnv(
        X_seq=Xte,
        y_seq=yte,
        files_seq=fte,
        detector_ckpt=DETECTOR_CKPT,
        scaler_ckpt=SCALER_CKPT,
        detector_type=DETECTOR_TYPE,
        device=DEVICE,
        max_steps_per_episode=MAX_STEPS_EP,
    )

    q = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
    q.load_state_dict(torch.load(RL_MODEL, map_location=DEVICE))
    q.eval()
    # load the trained rl model
    # and set to inference mode

    total_reward = 0.0
    n_steps = 0

    TP = TN = FP = FN = 0
    action_counts = Counter()

    #list to collect per-step inference latency in milliseconds
    step_latencies_ms = []

    episodes = 30
    for _ in range(episodes):
        s = env.reset()
        done = False

        while not done:
            # start the clock just before inference
            # this measures how long it takes the RL agent to decide
            # on an action given a sequence of 20 packets (one state vector)
            t_start = time.perf_counter()

            with torch.no_grad():
                qs = q(
                    torch.tensor(s, dtype=torch.float32, device=DEVICE)
                    .unsqueeze(0)
                )
                a = int(torch.argmax(qs, dim=1).item())

            #stop the clock right after the action is selected
            t_end = time.perf_counter()

            #convert to milliseconds and record
            latency_ms = (t_end - t_start) * 1000.0
            step_latencies_ms.append(latency_ms)

            ns, r, done, info = env.step(a)
            s = ns

            total_reward += r
            n_steps += 1

            action_counts[info["action_name"]] += 1

            y = int(info["y_true"])
            act = int(info["action"])

            is_allow = (act == A_ALLOW)
            is_mitigate = (act not in NON_MITIGATE_ACTIONS)

            if y == 1:  # attack
                if is_mitigate:
                    TP += 1
                else:
                    FN += 1
            else:  # normal
                if is_allow:
                    TN += 1
                else:
                    FP += 1

    attack_total = TP + FN
    normal_total = TN + FP

    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    TPR = TP / max(attack_total, 1)
    TNR = TN / max(normal_total, 1)
    FPR = FP / max(normal_total, 1)
    FNR = FN / max(attack_total, 1)

    #compute response time statistics from collected latencies
    latencies_arr = np.array(step_latencies_ms)
    avg_latency_ms   = float(np.mean(latencies_arr))
    min_latency_ms   = float(np.min(latencies_arr))
    max_latency_ms   = float(np.max(latencies_arr))
    p95_latency_ms   = float(np.percentile(latencies_arr, 95))
    throughput_per_s = 1000.0 / avg_latency_ms  #steps (decisions) per second

    # now we will only write the results
    print(f"\n=== RL TEST METRICS FOR {detector_name} (TRUE MITIGATION) ===")
    print("Episodes:", episodes)  #tills how many test episodes (files) were evaluated
    print("Steps:", n_steps)  #says the total number of actions taken by the rl agent
    print("Avg reward per step:", total_reward / max(n_steps, 1))
    # this tells on average were actions good or bad

    print("\nConfusion-style counts:")
    print("TP (attack mitigated):", TP)
    print("FN (attack missed):", FN)
    print("TN (normal allowed):", TN)
    print("FP (normal blocked):", FP)

    print("\nRates:")
    print(f"TPR (Attack Mitigation): {TPR:.4f}")
    print(f"TNR (Normal Allow):     {TNR:.4f}")
    print(f"FPR (False Positive):   {FPR:.4f}")
    print(f"FNR (False Negative):   {FNR:.4f}")

    print("\nClassification metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nSafety verdict:")
    if (FPR <= MAX_FPR) and (FNR <= MAX_FNR):
        verdict = "ACCEPTABLE POLICY (≤ 5% FPR & FNR)"
        print(verdict)
    else:
        verdict = "REJECTED POLICY (violates safety bounds)"
        print(verdict)

    print("\nTop actions:")
    for name, c in action_counts.most_common(12):
        print(f"  {name:18s} : {c}")

    #print response time results — this tells us how fast the system
    # can make a real-time decision after receiving a full sequence of 20 packets
    print("\nResponse Time (per sequence of 20 packets → 1 decision):")
    print(f"  Avg latency : {avg_latency_ms:.4f} ms")
    print(f"  Min latency : {min_latency_ms:.4f} ms")
    print(f"  Max latency : {max_latency_ms:.4f} ms")
    print(f"  P95 latency : {p95_latency_ms:.4f} ms  ← real-time guarantee threshold")
    print(f"  Throughput  : {throughput_per_s:.1f} decisions/second")

    # return summary for comparison
    return {
        "detector_name": detector_name,
        "TPR": TPR,
        "TNR": TNR,
        "FPR": FPR,
        "FNR": FNR,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_reward": total_reward / max(n_steps, 1),
        "verdict": verdict,
        # ← ADDED: include timing metrics in the returned summary
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "throughput_per_s": throughput_per_s,
    }


# now we will test and evaluate
def main():
    print("[INFO] Building TEST sequences...")

    data = build_rl_data_from_csv(
        CSV_ALL, TRAIN_IDX, VAL_IDX, TEST_IDX,
        seq_len=SEQ_LEN,
        step=STEP
    )

    # this will store results for all detectors
    all_results = []

    #now evaluating all models
    for detector_name, detector_config in DETECTOR_CONFIGS.items():
        result = evaluate_single_detector(detector_name, detector_config, data)
        if result is not None:
            all_results.append(result)

    #final comparsion noww
    if len(all_results) > 0:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON OF ALL DETECTORS")
        print("=" * 80)

        for result in all_results:
            print(f"\n{result['detector_name']}:")
            print(f"  TPR: {result['TPR']:.4f} | TNR: {result['TNR']:.4f}")
            print(f"  FPR: {result['FPR']:.4f} | FNR: {result['FNR']:.4f}")
            print(f"  F1: {result['f1']:.4f} | Avg Reward: {result['avg_reward']:.4f}")
            print(f"  Verdict: {result['verdict']}")
            # ← ADDED: also show timing in the final comparison table
            print(f"  Avg Latency: {result['avg_latency_ms']:.4f} ms | P95: {result['p95_latency_ms']:.4f} ms | Throughput: {result['throughput_per_s']:.1f} dec/s")

        # find the best detector by f1 score
        best_detector = max(all_results, key=lambda x: x['f1'])
        print(f"\nBEST DETECTOR BY F1: {best_detector['detector_name']} with F1={best_detector['f1']:.4f}")

        # ← ADDED: also report the fastest detector by average latency
        fastest_detector = min(all_results, key=lambda x: x['avg_latency_ms'])
        print(f"FASTEST DETECTOR   : {fastest_detector['detector_name']} with Avg Latency={fastest_detector['avg_latency_ms']:.4f} ms")


if __name__ == "__main__":
    main()