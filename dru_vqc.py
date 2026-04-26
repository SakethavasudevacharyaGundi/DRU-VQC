import os
import json
import random
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

SEED = 42
EPOCHS = 50
LR = 0.03
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10
EPS = 1e-8

BASE_DIR = "."
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_CSV = os.path.join(OUTPUT_DIR, "step3_train_800.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "step3_test_200.csv")
TOP12_JSON = os.path.join(OUTPUT_DIR, "mi_top12_batches.json")

LOSS_CSV = os.path.join(OUTPUT_DIR, "dru_vqc_loss_curve.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "dru_vqc_metrics.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "dru_vqc_metrics.json")
LEARNING_CURVE_CSV = os.path.join(OUTPUT_DIR, "dru_vqc_learning_curve.csv")
MODEL_PARAMS_NPY = os.path.join(OUTPUT_DIR, "dru_vqc_theta.npy")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def sigmoid(x):
    return 1.0 / (1.0 + qml.numpy.exp(-x))

def bce_loss(y_true, y_prob):
    p = np.clip(y_prob, EPS, 1.0 - EPS)
    return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    with open(TOP12_JSON, "r") as f:
        top12 = json.load(f)

    # Sort features by rank to maintain consistency
    top12 = sorted(top12, key=lambda x: x["rank"])
    features = [item["feature"] for item in top12]

    # Reconstruct binary label if missing
    for df in [train_df, test_df]:
        if "binary_label" not in df.columns:
            df["binary_label"] = (df["source_class"] != "benign").astype(int)

    X_train_raw = train_df[features].to_numpy(dtype=float)
    y_train = train_df["binary_label"].to_numpy(dtype=int)

    X_test_raw = test_df[features].to_numpy(dtype=float)
    y_test = test_df["binary_label"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, y_train, X_test, y_test, features

dev = qml.device("lightning.qubit", wires=4)

@qml.qnode(dev, diff_method="adjoint")
def circuit(x, theta):
    # x is of shape (12,)
    # theta is of shape (3, 4)
    for block in range(3):
        # Ry(feature) encoding for the 4 features in this batch
        for w in range(4):
            qml.RY(x[block * 4 + w], wires=w)

        # CNOT chain
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])

        # Variational layer
        for w in range(4):
            qml.RY(theta[block, w], wires=w)

    # Measure same as ablation
    # myPart.md text says "(Z0+Z1+Z2+Z3)/4", but ablation_vqc.py uses tensor product.
    # We will use the average of Z operators as explicitly requested in the text.
    # We'll use qml.math.sum if possible, or return list and average later.
    # To keep the qnode returning a single scalar differentiable natively,
    # we can use qml.Hamiltonian:
    obs = qml.Hamiltonian([0.25, 0.25, 0.25, 0.25], [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2), qml.PauliZ(3)])
    return qml.expval(obs)

def predict_scores(X, theta):
    return np.array([circuit(x, theta) for x in X], dtype=float)

def predict_proba(X, theta):
    scores = predict_scores(X, theta)
    return sigmoid(scores)

def train_model(X_train, y_train, X_val, y_val):
    theta = qml.numpy.array(np.random.uniform(-0.1, 0.1, size=(3, 4)), requires_grad=True)
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

        history.append({
            "epoch": epoch,
            "batch_loss_mean": mean_batch_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_theta = theta.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:02d} | batch_loss={mean_batch_loss:.6f} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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
    X_train_full, y_train_full, X_test, y_test, features = load_data()
    X_train, y_train, X_val, y_val = stratified_train_val_split(X_train_full, y_train_full, val_fraction=0.2)

    print("Training DRU-VQC Full Model...")
    theta_star, history = train_model(X_train, y_train, X_val, y_val)

    # Save the parameters
    np.save(MODEL_PARAMS_NPY, theta_star)

    train_metrics = evaluate(X_train_full, y_train_full, theta_star)
    test_metrics = evaluate(X_test, y_test, theta_star)

    loss_df = pd.DataFrame(history)
    loss_df.to_csv(LOSS_CSV, index=False)

    metrics_rows = [
        {"model": "DRU_VQC", "split": "train_full", **train_metrics},
        {"model": "DRU_VQC", "split": "test", **test_metrics},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    # Proposed learning curve (10%, 25%, 50%, 100%)
    fractions = [0.1, 0.25, 0.5, 1.0]
    curve_rows = []
    print("Computing learning curve...")
    for frac in fractions:
        if frac == 1.0:
            curve_rows.append({
                "fraction": 1.0,
                "train_size": len(y_train_full),
                "f1": test_metrics["f1"]
            })
            continue

        rng = np.random.default_rng(SEED)
        idx0 = np.where(y_train_full == 0)[0]
        idx1 = np.where(y_train_full == 1)[0]
        rng.shuffle(idx0)
        rng.shuffle(idx1)

        n0 = max(1, int(round(len(idx0) * frac)))
        n1 = max(1, int(round(len(idx1) * frac)))

        sub_idx = np.concatenate([idx0[:n0], idx1[:n1]])
        rng.shuffle(sub_idx)

        X_sub = X_train_full[sub_idx]
        y_sub = y_train_full[sub_idx]

        X_sub_train, y_sub_train, X_sub_val, y_sub_val = stratified_train_val_split(X_sub, y_sub, val_fraction=0.2)

        theta_sub, _ = train_model(X_sub_train, y_sub_train, X_sub_val, y_sub_val)
        sub_test_metrics = evaluate(X_test, y_test, theta_sub)

        curve_rows.append({
            "fraction": frac,
            "train_size": len(sub_idx),
            "f1": sub_test_metrics["f1"]
        })

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(LEARNING_CURVE_CSV, index=False)

    payload = {
        "seed": SEED,
        "epochs_max": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "features": features,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_theta": [float(v) for v in np.array(theta_star).flatten()],
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("DRU-VQC training complete.")
    print(metrics_df.to_string(index=False))
    print(curve_df.to_string(index=False))

if __name__ == "__main__":
    main()
