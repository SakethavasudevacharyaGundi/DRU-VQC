import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.preprocessing import StandardScaler
import json

BASE_DIR = "."
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TRAIN_CSV = os.path.join(OUTPUT_DIR, "step3_train_800.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "step3_test_200.csv")
TOP12_JSON = os.path.join(OUTPUT_DIR, "mi_top12_batches.json")
MODEL_PARAMS_NPY = os.path.join(OUTPUT_DIR, "dru_vqc_theta.npy")
SHAP_PLOT_PNG = os.path.join(OUTPUT_DIR, "dru_vqc_shap_bar.png")

SEED = 42

def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    with open(TOP12_JSON, "r") as f:
        top12 = json.load(f)

    top12 = sorted(top12, key=lambda x: x["rank"])
    features = [item["feature"] for item in top12]

    X_train_raw = train_df[features].to_numpy(dtype=float)
    X_test_raw = test_df[features].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, features

dev = qml.device("lightning.qubit", wires=4)

@qml.qnode(dev)
def circuit(x, theta):
    for block in range(3):
        for w in range(4):
            qml.RY(x[block * 4 + w], wires=w)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        for w in range(4):
            qml.RY(theta[block, w], wires=w)
            
    obs = qml.Hamiltonian([0.25, 0.25, 0.25, 0.25], [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2), qml.PauliZ(3)])
    return qml.expval(obs)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    if not os.path.exists(MODEL_PARAMS_NPY):
        print(f"Error: {MODEL_PARAMS_NPY} not found. Run dru_vqc.py first.")
        return

    theta_star = np.load(MODEL_PARAMS_NPY)
    X_train, X_test, features = load_data()

    def model_predict(X):
        # SHAP passes a 2D numpy array
        scores = np.array([circuit(x, theta_star) for x in X])
        return sigmoid(scores)

    rng = np.random.default_rng(SEED)
    
    # 20 background training samples
    idx_train = rng.choice(len(X_train), size=20, replace=False)
    background = X_train[idx_train]

    # 30 explainer test samples
    idx_test = rng.choice(len(X_test), size=30, replace=False)
    test_samples = X_test[idx_test]

    print("Running SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_samples, nsamples=100)

    print("Generating SHAP bar plot...")
    plt.figure()
    shap.summary_plot(shap_values, test_samples, feature_names=features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PNG, dpi=300)
    print(f"Saved SHAP plot to {SHAP_PLOT_PNG}")

if __name__ == "__main__":
    main()
