#5 runn
import random
import csv
import numpy as np
import torch
from torch import nn
from pathlib import Path
from collections import Counter
import argparse
# now we will import:
# the environment where RL interacts
# and the data builder to make sequences
# and the actions names & action ids
from rl_env import (
    NeuroGuardRLEnv,
    build_rl_data_from_csv,
    ACTION_NAMES,
    A_ALLOW,
    A_ALERT_ONLY,
    A_ESCALATE,
    A_DEESCALATE,
)

NON_MITIGATE_ACTIONS = {A_ALLOW, A_ALERT_ONLY, A_ESCALATE, A_DEESCALATE}
# these actions do not really stop an attack
# so during evaluation:
# if attack and agent chooses any of these then count it as fn
# only other actions count as mitigation will be counted as tp

OUT_DIR = Path(r"C:\Users\User\Documents\defender_data_tcp_fixed")  # this is the main folder that contains csv, indices and saved models

CSV_ALL = OUT_DIR / "mqtt_packets_labeled.csv"  # the full dataset
# those are the saved row indices for train/val/test split
TRAIN_IDX = OUT_DIR / "train_idx.npy"
VAL_IDX = OUT_DIR / "val_idx.npy"
TEST_IDX = OUT_DIR / "test_idx.npy"

#
# this dictionary maps detector names to their checkpoint files
DETECTOR_CONFIGS = {
    "cnn_only": {
        "ckpt": OUT_DIR / "detector_cnn_only.pt",
        "scaler": OUT_DIR / "scaler_cnn_only.pkl",
        "type": "cnn_only"
    },
    "cnn_attn": {
        "ckpt": OUT_DIR / "detector_cnn_attention.pt",
        "scaler": OUT_DIR / "scaler_cnn_attention.pkl",
        "type": "cnn_attention"
    },
    "cnn_bilstm_attn": {
        "ckpt": OUT_DIR / "detector_cnn_bilstm_attn.pt",
        "scaler": OUT_DIR / "scaler_cnn_bilstm_attn.pkl",
        "type": "cnn_bilstm_attn"
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 20
STEP = 5
# the sequence window=20 packets and slide by 5 packets
N_ACTIONS = 16
STATE_DIM = 16
# the rl agent output has 16 actions
# and the rl state vector has 16 numbers
EPISODES = 225
MAX_STEPS_EP = 200
# this means train for 200 episodes and each episode max 200 steps
GAMMA = 0.99
# now gamma controls how much the agent cares about the future
# 0.99 means future rewards matter a lot
LR = 1e-3
BATCH = 256
# means that the agent learns from 256 past experiences at once

REPLAY_SIZE = 200_000
# here the agent remembers up to 200,000 past experiences(it can store)

START_LEARNING = 5_000
# the agent doesn't start learning immediately
# it first collects 5,000 experiences

TARGET_UPDATE = 1_000
# and here how often to refresh target network

EPS_START = 1.0  # this means to start with fully random actions
EPS_END = 0.05  # end with mostly the highest learned reward actions
EPS_DECAY_STEPS = 50_000  # linearly decay epsilon over 50k steps
# so we slowly reduce randomness so the agent explores early

# we evaluate on 30 validation episodes each time (more stable)
VAL_EVAL_EPISODES = 30
VAL_SEED = 123  # fixed seed so validation episodes are the same every time
# this makes comparisons fair

MAX_FPR = 0.05
MAX_FNR = 0.01
MIN_TPR = 0.99
MIN_TNR = 0.95
# model must satisfy these to be accepted because this is my target
# else it will be rejected even if reward is high


# now we will write a function to log training metrics into a csv file
# so you can analyze results later (Excel, Python, plots)
def append_csv_row(csv_path: Path, header, row_dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # to make sure folder exists
    file_exists = csv_path.exists()  # and here check if csv already exists
    with open(csv_path, "a", newline="", encoding="utf-8") as f:  # now open csv in append mode don't overwrite
        w = csv.DictWriter(f, fieldnames=header)  # and this means to write rows using column names
        if not file_exists:
            w.writeheader()
            # now if the csv file doesn't exist yet
            # we will write the column names the header
            # this happens only once
        w.writerow(row_dict)
        # now we will write one row
        # and that row contains metrics of one episode
        # and this runs every episode


# now this function is about the Q-network it's the agent's brain
# it takes the current state
# and outputs a score (Q-value) for every possible action
# a higher Q-value means a better action
# agent chooses the action with the highest Q-value
# so it maps the state to Q-values for all actions
class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()  # as before
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            # the input is the state (16 values)
            # expand to 256 neurons so we can have more capacity to learn patterns
            nn.ReLU(),  # it adds a non-linearity to allow complex behavior
            nn.Linear(256, 256),
            nn.ReLU(),
            # then two same layers so a deeper network lead to a better decision making
            nn.Linear(256, n_actions),
            # output one Q-value per action
            # and the agent picks action with highest Q
            # because Q-values is about how good is this action in this state
        )

    def forward(self, x):
        return self.net(x)
        # this defines how data flows through the network
        # x is the input state (16 numbers)
        # self.net(x) will pass the input through all layers


# the replay buffer is the memory
# so this will store the past experiences so learning is stable
class Replay:
    def __init__(self, cap):
        self.cap = cap  # cap is the maximum number of experiences it can store
        self.s = np.zeros((cap, STATE_DIM), dtype=np.float32)
        # this stores the states so it's shape is cap * state_size (200k × 16)
        self.a = np.zeros((cap,), dtype=np.int64)  # this store actions taken by the agent
        self.r = np.zeros((cap,), dtype=np.float32)  # and this store the rewards received
        self.ns = np.zeros((cap, STATE_DIM), dtype=np.float32)  # and this stores the next states (state after action)
        self.d = np.zeros((cap,), dtype=np.float32)  # and this stores the done flags
        # as 1 means episode ended and 0 means still running
        # so in short state, action, reward, next state, done
        self.i = 0  # this is a pointer that says where to write the next experience
        self.n = 0  # and this says how many experiences are currently stored

    # now this function is to add one experience to memory
    # the experience = (state, action, reward, next state, done)
    def add(self, s, a, r, ns, d):
        self.s[self.i] = s
        self.a[self.i] = a
        self.r[self.i] = r
        self.ns[self.i] = ns
        self.d[self.i] = float(d)
        # these as before
        self.i = (self.i + 1) % self.cap  # now this means to move the pointer forward
        # when the buffer is full then overwrite old data
        self.n = min(self.n + 1, self.cap)  # this increase the stored count
        # and never exceed the buffer size

    def sample(self, batch):  # now this will ask for a random batch of experiences
        idx = np.random.randint(0, self.n, size=batch)  # so pick a random memory positions
        # now this will return a batch of:
        # states, actions, rewards, next states, done flags then convert them to PyTorch tensors for training
        return (
            torch.tensor(self.s[idx]),
            torch.tensor(self.a[idx]),
            torch.tensor(self.r[idx]),
            torch.tensor(self.ns[idx]),
            torch.tensor(self.d[idx]),
        )
    # leakage is ensured here because we randomly pick exp
    # only from the replay buffer and the replay buffer contains train data only


# now this function tells us how much randomness (epsilon:randomness in action choice) to use at a given step
def eps_by_step(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
        # it means that if we are past the decay period
        # then stop changing epsilon and use the final small randomness (EPS_END)
    frac = step / EPS_DECAY_STEPS
    # we divide step by EPS_DECAY_STEPS to know how far training has progressed and reduce randomness smoothly
    return EPS_START + (EPS_END - EPS_START) * frac  # this return will slowly move epsilon from:
    # EPS_START (very random)
    # to EPS_END (mostly greedy)
    # this is a smooth and a straight-line decrease


# this function evaluates the rl agent to measure the performance
def evaluate_greedy_true_metrics(env, qnet, episodes=5, seed=None):
    # this means evaluate trained rl policy using greedy actions only
    # env means the rl environment
    # qnet the trained Q-network (the agent's brain)
    # episo means how many episodes (files) to evaluate, this is like check a few files quickly to see if learning is improving
    # none means random each time
    qnet.eval()  # means to switch to evaluation mode
    total_reward = 0.0  # now we start a counter at zero

    TP = TN = FP = FN = 0  # and the counters for attack/normal
    action_counts = Counter()  # and here we will count how often each action is chosen

    # we will do this so validation results are fair and comparable across episodes
    # so we make validation fair by testing the model the same way every time
    if seed is not None:
        # we are saving the current random state
        # so we can restore it later
        rng_state = np.random.get_state()
        py_state = random.getstate()
        # and now fix randomness
        # validation will use the same files and order every time
        np.random.seed(seed)
        random.seed(seed)

    try:
        for _ in range(episodes):  # now loop over validation episodes (files)
            s = env.reset()  # and now start new episode and get initial state
            done = False
            while not done:  # means keep going until this file is finished
                with torch.no_grad():  # means no learning inference only
                    qs = qnet(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))  # now get the Q-values for all actions
                    a = int(torch.argmax(qs, dim=1).item())  # and pick the best (greedy) action

                ns, r, done, info = env.step(a)  # now apply the action, get result
                s = ns  # now move to the next state then continue from there
                total_reward += r  # now add this step's reward to the total score

                action_counts[info["action_name"]] += 1  # now we'll count how many times each action is used

                y = int(info["y_true"])  # 1 attack and 0 normal
                act = int(info["action"])

                is_allow = (act == A_ALLOW)
                is_mitigate = (act not in NON_MITIGATE_ACTIONS)
                # if attack and mitigated then TP
                # if attack and not mitigated then FN
                # if normal and allowed then TN
                # if normal and blocked then FP
                if y == 1:
                    if is_mitigate:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if is_allow:
                        TN += 1
                    else:
                        FP += 1
    finally:  # now this block always runs even if something went wrong
        if seed is not None:
            # now restore randomness back to normal
            # so training is not affected by validation
            np.random.set_state(rng_state)
            random.setstate(py_state)

    attack_total = TP + FN
    normal_total = TN + FP

    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)  # same as TPR
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # training the rl learns from reward but validation metrics tell you if it actually works that's why we did them only here in val
    # cuz the agent is exploring (random actions, epsilon)
    # so accuracy/recall during training are noisy
    # and reward is the main learning signal
    # now calculate the security metrics
    tpr = TP / max(attack_total, 1)
    tnr = TN / max(normal_total, 1)
    fpr = FP / max(normal_total, 1)
    fnr = FN / max(attack_total, 1)

    qnet.train()  # and now switch back to training mode
    return {
        "val_reward": float(total_reward / max(episodes, 1)),
        "TP": int(TP), "FN": int(FN), "TN": int(TN), "FP": int(FP),
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "TPR": float(tpr), "TNR": float(tnr), "FPR": float(fpr), "FNR": float(fnr),
        "top_actions": action_counts,
    }


def save_top3_checkpoints(q, score, ep, top3, paths):  # now this function aims to keep only the best 3 rl models
    cand = (float(score), int(ep), None)
    # this means create one record for this model:
    # score tells how good the model is
    # ep is which episode produced it
    # none for the file path not assigned yet

    top3.append(cand)  # add this model's record to the list of best models
    top3.sort(key=lambda x: x[0], reverse=True)  # sort models by score (best first)
    top3 = top3[:3]  # and keep only the top 3 models

    # those are the file paths for best, 2nd best, 3rd best models
    for i, (sc, e, _) in enumerate(top3):
        torch.save(q.state_dict(), paths[i])  # and loop over the top 3 models
        # then save model weights to disk
        top3[i] = (sc, e, str(paths[i]))  # then update entry with saved file path

    return top3


def is_acceptable(vm):  # this to check if a model meets security requirements i made
    return (
            (vm["FPR"] <= MAX_FPR) and
            (vm["FNR"] <= MAX_FNR) and
            (vm["TPR"] >= MIN_TPR) and
            (vm["TNR"] >= MIN_TNR)  # i did and wrote the limits before
    )


# now this function trains RL for one detector
def train_single_detector(detector_name, detector_config, data):
    print(f"\n{'=' * 80}")
    print(f"[INFO] Starting RL training for detector: {detector_name}")
    print(f"{'=' * 80}\n")

    # get the detector checkpoint and scaler paths
    DETECTOR_CKPT = detector_config["ckpt"]
    SCALER_CKPT = detector_config["scaler"]
    DETECTOR_TYPE = detector_config["type"]

    # create output file paths specific to this detector
    RL_BEST1 = OUT_DIR / f"rl_dqn_preventer_best_{detector_name}.pt"
    RL_BEST2 = OUT_DIR / f"rl_dqn_preventer_rank2_{detector_name}.pt"
    RL_BEST3 = OUT_DIR / f"rl_dqn_preventer_rank3_{detector_name}.pt"
    RL_ACCEPTED_BEST = OUT_DIR / f"rl_policy_best_ACCEPTED_{detector_name}.pt"
    RL_OVERALL_BEST = OUT_DIR / f"rl_policy_best_OVERALL_{detector_name}.pt"
    RL_LOG_CSV = OUT_DIR / f"rl_training_metrics_{detector_name}.csv"

    # get train and validation data
    Xtr, ytr, ftr = data["train"]["X"], data["train"]["y"], data["train"]["files"]
    Xva, yva, fva = data["val"]["X"], data["val"]["y"], data["val"]["files"]

    # create training environment
    env = NeuroGuardRLEnv(  # the environment used for rl learning
        X_seq=Xtr, y_seq=ytr, files_seq=ftr,
        detector_ckpt=DETECTOR_CKPT,
        scaler_ckpt=SCALER_CKPT,
        detector_type=DETECTOR_TYPE,
        device=DEVICE,
        max_steps_per_episode=MAX_STEPS_EP,
    )

    # create validation environment
    val_env = NeuroGuardRLEnv(  # and here we separate environment used only for validation
        X_seq=Xva, y_seq=yva, files_seq=fva,
        detector_ckpt=DETECTOR_CKPT,
        scaler_ckpt=SCALER_CKPT,
        detector_type=DETECTOR_TYPE,
        device=DEVICE,
        max_steps_per_episode=MAX_STEPS_EP,
    )

    q = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
    # this is the brain that learns
    # it is updated every training step
    tq = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
    # this is a frozen copy of the brain
    # it is used only to compute stable targets so  make learnig stable
    tq.load_state_dict(q.state_dict())
    # this copies all weights from q into tq
    # at the start the target network must match the main network otherwise targets would be random and wrong
    # so we start both networks equal then update target slowly to stabilize training
    opt = torch.optim.Adam(q.parameters(), lr=LR)
    # the optimizer decides how to change the network weights to learn
    loss_fn = nn.SmoothL1Loss()  # and the loss function measures how wrong the Q-values are
    # we need it because DQN learns by:
    # predicting a Q-value
    # then comparing it with the target Q-value
    # then reducing the difference
    # so the loss measures the difference
    # and the optimizer fixes it

    replay = Replay(REPLAY_SIZE)  # a memory to store experiences (s,a,r,ns,done)

    global_step = 0  # this says the total number of actions taken so far it increases every step
    # across all episodes we need it to control epsilon decay and target update
    # and this will keep the 3 best checkpoints as backup
    top3 = []

    best_overall_score = -1e9
    # we will start with a very small number so any real model score will be better
    # it's used to track the best model overall
    best_accepted_score = -1e9
    # same idea to track the best model that meets safety thresholds
    best_accepted_ep = None  # to remember which episode produced the accepted best model
    # the column names for the csv we created for logging
    header = [
        "episode", "train_reward", "val_reward", "eps", "global_step",
        "TPR", "TNR", "FPR", "FNR", "TP", "TN", "FP", "FN",
        "val_score", "accepted", "accepted_score",
        "top1_action", "top1_count", "top2_action", "top2_count", "top3_action", "top3_count"
    ]
    if RL_LOG_CSV.exists():  # this means delete the old log file and start fresh
        RL_LOG_CSV.unlink()

    print(f"[INFO] Start DQN training for {detector_name}...")
    for ep in range(1, EPISODES + 1):  # train for many episodes and each episode = one file of sequences
        s = env.reset()
        ep_reward = 0.0
        done = False  # means start new episode with reward = 0

        while not done:  # this will allow it to run until episode ends, either file finished or max steps to prevent very long files from dominating training
            eps = eps_by_step(global_step)  # the current randomness rate

            if random.random() < eps:
                a = random.randint(0, N_ACTIONS - 1)
                # this means to explore early, exploit later
                # early training tries many actions while later training uses the best ones

            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                    a = int(torch.argmax(qs, dim=1).item())
                    # now here the network only predicts
                    # it does not learn here so no gradients are stored and we do this because
                    # action selection is not training and it saves memory and speed and revents accidental weight updates

            ns, r, done, _ = env.step(a)  # now apply action then get next state + reward + done

            replay.add(s, a, r, ns, done)
            # now we store experience in replay buffer to be able to remember past actions and
            # outcomes so the agent can learn from old experiences not just the latest one
            s = ns
            ep_reward += r
            global_step += 1
            # now move forward, collect reward and update step counter

            if replay.n >= START_LEARNING:  # this means don't train too early (need enough experiences)
                bs, ba, br, bns, bd = replay.sample(BATCH)  # this means that random batches are
                # taken from the replay buffer filled by past agent actions

                bs = bs.to(DEVICE)
                ba = ba.to(DEVICE)
                br = br.to(DEVICE)
                bns = bns.to(DEVICE)
                bd = bd.to(DEVICE)  # cpu

                qsa = q(bs).gather(1, ba.view(-1, 1)).squeeze(1)
                # q(bs) this gives Q-values for all actions
                # .gather(...) selects only the action that was actually taken
                with torch.no_grad():  # not learning
                    best_a = torch.argmax(q(bns), dim=1)
                    # it means use main network (q)
                    # pick the best next action
                    next_q = tq(bns).gather(1, best_a.view(-1, 1)).squeeze(1)
                    # use target network (tq)
                    # and get the Q-value of that best action
                    target = br + GAMMA * (1.0 - bd) * next_q
                    # the final learning target:
                    # r is the immediate reward
                    # gamma*next_q means the future reward
                    # (1-done) means no future reward if episode ended

                loss = loss_fn(qsa, target)
                # qsa means what the network predicted for the action it took
                # target means what the network should have predicted
                # the loss measures how wrong the prediction is

                opt.zero_grad()
                # means clear old gradients
                # aims to prevent mixing old mistakes with new ones
                loss.backward()
                # this computes how each weight caused the error
                opt.step()
                # this updates the network weights
                # and make the prediction less wrong next time

                if global_step % TARGET_UPDATE == 0:
                    tq.load_state_dict(q.state_dict())  # this updates the target network occasionally to keep learning stable

        # validation with true metrics
        vm = evaluate_greedy_true_metrics(val_env, q, episodes=VAL_EVAL_EPISODES, seed=VAL_SEED)
        # and this will let us get the current epsilon value
        # only for logging
        eps_now = eps_by_step(global_step)

        # now combine many metrics into one score so we can compare models easily
        val_score = (0.55 * vm["TPR"]) + (0.55 * vm["TNR"]) - (0.70 * vm["FPR"]) - (0.30 * vm["FNR"])

        # my aim is to meet my acceptable thresholds
        accepted = is_acceptable(vm)
        # This score is used only after a model already passed safety thresholds it ranks
        # safe models against each other
        accepted_score = (vm["TPR"] + vm["TNR"]) - (vm["FPR"] + vm["FNR"])

        tops = vm["top_actions"].most_common(3)  # we will get the 3 most used actions during validation
        while len(tops) < 3:
            tops.append(("NONE", 0))
            # the base rule
            # so to make sure we always have 3 entries and avoid errors if fewer actions were used

        status = "ACCEPT" if accepted else "REJECT"
        # label model:ACCEPT means it meets safety thresholds and REJECT means it fails thresholds
        print(
            f"[{detector_name}] EP {ep:03d} | train_reward={ep_reward:.2f} | val_reward={vm['val_reward']:.2f} "
            f"| eps={eps_now:.3f} | TPR={vm['TPR']:.3f} TNR={vm['TNR']:.3f} "
            f"FPR={vm['FPR']:.3f} FNR={vm['FNR']:.3f} | score={val_score:.4f} | {status} | top={tops[0][0]}"
        )

        append_csv_row(RL_LOG_CSV, header, {
            "episode": ep,
            "train_reward": float(ep_reward),
            "val_reward": float(vm["val_reward"]),
            "eps": float(eps_now),
            "global_step": int(global_step),

            "TPR": float(vm["TPR"]),
            "TNR": float(vm["TNR"]),
            "FPR": float(vm["FPR"]),
            "FNR": float(vm["FNR"]),
            "TP": int(vm["TP"]),
            "TN": int(vm["TN"]),
            "FP": int(vm["FP"]),
            "FN": int(vm["FN"]),

            "val_score": float(val_score),
            "accepted": int(accepted),
            "accepted_score": float(accepted_score),

            "top1_action": str(tops[0][0]),
            "top1_count": int(tops[0][1]),
            "top2_action": str(tops[1][0]),
            "top2_count": int(tops[1][1]),
            "top3_action": str(tops[2][0]),
            "top3_count": int(tops[2][1]),
        })

        # always save the best overall (backup)
        if val_score > best_overall_score:
            best_overall_score = val_score
            torch.save(q.state_dict(), RL_OVERALL_BEST)
            print(f"  [OK] saved best overall: {RL_OVERALL_BEST} | score={best_overall_score:.4f}")

        # and save the best accepted (only if meets my acceptable thresholds)
        if accepted and (accepted_score > best_accepted_score):
            best_accepted_score = accepted_score
            best_accepted_ep = ep
            torch.save(q.state_dict(), RL_ACCEPTED_BEST)

            print(
                f" [ACCEPTED] saved best accepted: {RL_ACCEPTED_BEST} | ep={ep} "
                f"| TPR={vm['TPR']:.3f} TNR={vm['TNR']:.3f} FPR={vm['FPR']:.3f} FNR={vm['FNR']:.3f} "
                f"| accepted_score={best_accepted_score:.4f}"
            )
        elif not accepted:
            print(
                f" not accepted (need: TPR≥{MIN_TPR}, TNR≥{MIN_TNR}, FPR≤{MAX_FPR}, FNR≤{MAX_FNR})"
            )

        # and we will keep top-3 checkpoints
        paths = [RL_BEST1, RL_BEST2, RL_BEST3]
        if (len(top3) < 3) or (val_score > top3[-1][0]):
            top3 = save_top3_checkpoints(q, val_score, ep, top3, paths)
            print(f"  [OK] updated Top-3 (score-based) checkpoints for {detector_name}:")
            for i, (sc, e, p) in enumerate(top3, 1):
                print(f"    rank{i}: score={sc:.4f} ep={e} file={p}")

    print(f"\n[DONE] RL training finished for {detector_name}.")
    print("Saved training log CSV:", RL_LOG_CSV)

    print(f"\n=== FINAL SUMMARY FOR {detector_name} ===")
    print("Best Overall score:", best_overall_score)
    print("Saved:", RL_OVERALL_BEST)

    if best_accepted_ep is None:
        print("Best Accepted: None (no model met thresholds)")
        print(f"Thresholds were: TPR≥{MIN_TPR}, TNR≥{MIN_TNR}, FPR≤{MAX_FPR}, FNR≤{MAX_FNR}")
    else:
        print("Best ACCEPTED episode:", best_accepted_ep)
        print("Best ACCEPTED score:", best_accepted_score)
        print("Saved:", RL_ACCEPTED_BEST)

    print(f"\nTop-3 (score-based) checkpoints for {detector_name}:")
    for i, (sc, e, p) in enumerate(top3, 1):
        print(f"  rank{i}: score={sc:.4f} ep={e} file={p}")

    # return summary for comparison
    return {
        "detector_name": detector_name,
        "best_overall_score": best_overall_score,
        "best_accepted_score": best_accepted_score if best_accepted_ep else None,
        "best_accepted_ep": best_accepted_ep
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL with different detectors")
    parser.add_argument(
        "--detector",
        type=str,
        choices=["cnn_only", "cnn_attn", "cnn_bilstm_attn", "all"],
        default="all",
        help="Which detector to train with (default: all)"
    )
    args = parser.parse_args()

    print("[INFO] Building RL sequences...")
    data = build_rl_data_from_csv(
        CSV_ALL, TRAIN_IDX, VAL_IDX, TEST_IDX,
        seq_len=SEQ_LEN, step=STEP
    )

    # store results for all detectors
    all_results = []

    if args.detector == "all":
        # Train with all detectors
        for detector_name, detector_config in DETECTOR_CONFIGS.items():
            print("\n" + "=" * 80)
            print(f"[INFO] Training RL with detector: {detector_name}")
            print("=" * 80)
            result = train_single_detector(detector_name, detector_config, data)
            all_results.append(result)

        # Print final comparison
        print("\n" + "=" * 80)
        print("FINAL COMPARISON OF ALL DETECTORS")
        print("=" * 80)

        for result in all_results:
            print(f"\n{result['detector_name']}:")
            print(f"  Best Overall Score: {result['best_overall_score']:.4f}")
            if result['best_accepted_score'] is not None:
                print(f"  Best Accepted Score: {result['best_accepted_score']:.4f} (Episode {result['best_accepted_ep']})")
            else:
                print(f"  Best Accepted Score: None (no model met thresholds)")

        # find the best detector overall
        best_detector = max(all_results, key=lambda x: x['best_overall_score'])
        print(f"\nBEST DETECTOR OVERALL: {best_detector['detector_name']} with score {best_detector['best_overall_score']:.4f}")
    else:
        #here train with specific detector
        print("\n" + "=" * 80)
        print(f"[INFO] Training RL with detector: {args.detector}")
        print("=" * 80)
        result = train_single_detector(args.detector, DETECTOR_CONFIGS[args.detector], data)
        print(f"\n[DONE] Training completed for {args.detector}")
        print(f"Best Overall Score: {result['best_overall_score']:.4f}")
        if result['best_accepted_score'] is not None:
            print(f"Best Accepted Score: {result['best_accepted_score']:.4f} (Episode {result['best_accepted_ep']})")
        else:
            print(f"Best Accepted Score: None (no model met thresholds)")
