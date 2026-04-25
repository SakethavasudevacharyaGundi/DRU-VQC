import os
import json
import math
import random
import numpy as np
import pandas as pd
import pennylane as qml

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


SEED = 42
EPOCHS = 50
LR = 0.03
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10
EPS = 1e-8

BASE_DIR = r"c:\Users\saket\OneDrive\Documents\Quantum"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_CSV = os.path.join(OUTPUT_DIR, "pca_train_4d.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "pca_test_4d.csv")

LOSS_CSV = os.path.join(OUTPUT_DIR, "ablation_tensor_loss_curve.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "ablation_tensor_metrics.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "ablation_tensor_metrics.json")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "ablation_tensor_summary.json")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x):
    return 1.0 / (1.0 + qml.numpy.exp(-x))


def bce_loss(y_true, y_prob):
    p = np.clip(y_prob, EPS, 1.0 - EPS)
    return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def load_split(path: str):
    df = pd.read_csv(path)
    needed = ["pc1", "pc2", "pc3", "pc4", "binary_label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    X = df[["pc1", "pc2", "pc3", "pc4"]].to_numpy(dtype=float)
    y = df["binary_label"].astype(int).to_numpy()
    return X, y


dev = qml.device("lightning.qubit", wires=4)


@qml.qnode(dev, diff_method="parameter-shift")
def circuit(x, theta):
    qml.IQPEmbedding(x, wires=[0, 1, 2, 3])

    for w in range(4):
        qml.RY(theta[w], wires=w)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])

    obs = qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
    return qml.expval(obs)


def predict_scores(X, theta):
    return np.array([circuit(x, theta) for x in X], dtype=float)


def predict_proba(X, theta):
    scores = predict_scores(X, theta)
    return sigmoid(scores)


def train_model(X_train, y_train, X_val, y_val):
    theta = qml.numpy.array(np.random.uniform(-0.1, 0.1, size=(4,)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)

    history = []
    best_val = float("inf")
    best_theta = theta.copy()
    bad_epochs = 0

    n = len(X_train)

    def batch_loss(params, xb, yb):
        probs = qml.numpy.array([sigmoid(circuit(x, params)) for x in xb])
        p = qml.numpy.clip(probs, EPS, 1.0 - EPS)
        yb_q = qml.numpy.array(yb)
        return -qml.numpy.mean(yb_q * qml.numpy.log(p) + (1.0 - yb_q) * qml.numpy.log(1.0 - p))

    for epoch in range(1, EPOCHS + 1):
        idx = np.random.permutation(n)
        Xs = X_train[idx]
        ys = y_train[idx]

        batch_losses = []
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            xb = Xs[start:end]
            yb = ys[start:end]
            theta, loss_val = opt.step_and_cost(lambda p: batch_loss(p, xb, yb), theta)
            batch_losses.append(float(loss_val))

        train_probs = predict_proba(X_train, theta)
        val_probs = predict_proba(X_val, theta)

        train_loss = float(bce_loss(y_train, train_probs))
        val_loss = float(bce_loss(y_val, val_probs))
        mean_batch_loss = float(np.mean(batch_losses))

        history.append(
            {
                "epoch": epoch,
                "batch_loss_mean": mean_batch_loss,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_theta = theta.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:02d} | batch_loss={mean_batch_loss:.6f} "
            f"| train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience reached).")
            break

    return best_theta, history


def evaluate(X, y, theta):
    probs = predict_proba(X, theta)
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
    }


def stratified_train_val_split(X, y, val_fraction=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_val = max(1, int(round(len(idx0) * val_fraction)))
    n1_val = max(1, int(round(len(idx1) * val_fraction)))

    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    train_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Missing train file: {TRAIN_CSV}")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Missing test file: {TEST_CSV}")

    X_train_full, y_train_full = load_split(TRAIN_CSV)
    X_test, y_test = load_split(TEST_CSV)

    X_train, y_train, X_val, y_val = stratified_train_val_split(X_train_full, y_train_full, val_fraction=0.2)

    theta_star, history = train_model(X_train, y_train, X_val, y_val)

    train_metrics = evaluate(X_train_full, y_train_full, theta_star)
    test_metrics = evaluate(X_test, y_test, theta_star)

    loss_df = pd.DataFrame(history)
    loss_df.to_csv(LOSS_CSV, index=False)

    metrics_rows = [
        {"model": "Ablation_VQC_Tensor", "split": "train_full", **train_metrics},
        {"model": "Ablation_VQC_Tensor", "split": "test", **test_metrics},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    payload = {
        "seed": SEED,
        "epochs_max": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "diff_method": "parameter-shift",
        "readout": "expval(Z0 @ Z1 @ Z2 @ Z3)",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_theta": [float(v) for v in np.array(theta_star)],
        "loss_curve_file": LOSS_CSV,
        "metrics_csv": METRICS_CSV,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_shape": list(X_train_full.shape),
                "test_shape": list(X_test.shape),
                "loss_epochs_recorded": int(len(loss_df)),
                "outputs": {
                    "loss_csv": LOSS_CSV,
                    "metrics_csv": METRICS_CSV,
                    "metrics_json": METRICS_JSON,
                },
            },
            f,
            indent=2,
        )

    print("Ablation training complete.")
    print(metrics_df.to_string(index=False))
    print(f"Saved: {LOSS_CSV}")
    print(f"Saved: {METRICS_CSV}")
    print(f"Saved: {METRICS_JSON}")
    print(f"Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
