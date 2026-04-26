import os
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt

BASE_DIR = "."
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CIRCUIT_DIAGRAM_TXT = os.path.join(OUTPUT_DIR, "dru_vqc_circuit_diagram.txt")
COMPARISON_CSV = os.path.join(OUTPUT_DIR, "comparison_table.csv")
LEARNING_CURVE_PNG = os.path.join(OUTPUT_DIR, "learning_curve_comparison.png")

# Metric files
QSVM_METRICS = os.path.join(OUTPUT_DIR, "qsvm_metrics.csv")
ABLATION_METRICS = os.path.join(OUTPUT_DIR, "ablation_tensor_metrics.csv")
DRU_METRICS = os.path.join(OUTPUT_DIR, "dru_vqc_metrics.csv")
CLASSICAL_METRICS = os.path.join(OUTPUT_DIR, "classical_baselines_metrics.csv")

# Learning curve files
QSVM_LC = os.path.join(OUTPUT_DIR, "qsvm_learning_curve.csv")
DRU_LC = os.path.join(OUTPUT_DIR, "dru_vqc_learning_curve.csv")

def draw_circuit():
    import numpy as np
    dev = qml.device("default.qubit", wires=4)
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
        obs = qml.Hamiltonian([0.25]*4, [qml.PauliZ(i) for i in range(4)])
        return qml.expval(obs)

    x_dummy = np.random.uniform(0, 1, size=(12,))
    theta_dummy = np.random.uniform(0, 1, size=(3, 4))
    
    diagram = qml.draw(circuit)(x_dummy, theta_dummy)
    with open(CIRCUIT_DIAGRAM_TXT, "w") as f:
        f.write("DRU-VQC Proposed Circuit Diagram:\n\n")
        f.write(diagram)
    print(f"Saved circuit diagram to {CIRCUIT_DIAGRAM_TXT}")

def build_comparison_table():
    dfs = []
    
    # Load all metrics
    if os.path.exists(QSVM_METRICS):
        dfs.append(pd.read_csv(QSVM_METRICS))
    if os.path.exists(ABLATION_METRICS):
        dfs.append(pd.read_csv(ABLATION_METRICS))
    if os.path.exists(DRU_METRICS):
        dfs.append(pd.read_csv(DRU_METRICS))
    if os.path.exists(CLASSICAL_METRICS):
        dfs.append(pd.read_csv(CLASSICAL_METRICS))
        
    if not dfs:
        print("No metrics files found to build comparison table.")
        return
        
    combined = pd.concat(dfs, ignore_index=True)
    
    # Filter for test split
    test_only = combined[combined["split"] == "test"].copy()
    
    # Select columns
    cols = ["model", "accuracy", "f1", "precision", "recall"]
    final_table = test_only[cols]
    
    # Order rows according to step 7
    order = {
        "QSVM": 1,
        "Ablation_VQC_Tensor": 2,
        "DRU_VQC": 3,
        "SVC_RBF": 4,
        "Random_Forest": 5
    }
    final_table["order"] = final_table["model"].map(order)
    final_table = final_table.sort_values("order").drop(columns=["order"])
    
    final_table.to_csv(COMPARISON_CSV, index=False)
    print(f"Saved comparison table to {COMPARISON_CSV}")
    print("\nComparison Table:")
    print(final_table.to_string(index=False))
    
    # Extract Ablation and Proposed accuracies
    ablation_acc = final_table[final_table["model"] == "Ablation_VQC_Tensor"]["accuracy"]
    dru_acc = final_table[final_table["model"] == "DRU_VQC"]["accuracy"]
    if not ablation_acc.empty and not dru_acc.empty:
        delta = float(dru_acc.iloc[0]) - float(ablation_acc.iloc[0])
        print(f"\nAccuracy delta between proposed and ablation: {delta:.4f}")

def plot_learning_curve():
    plt.figure(figsize=(8, 6))
    
    if os.path.exists(QSVM_LC):
        q_df = pd.read_csv(QSVM_LC)
        plt.plot(q_df["fraction"] * 100, q_df["f1"], marker='s', linestyle='--', label="QSVM (2 points)")
        
    if os.path.exists(DRU_LC):
        d_df = pd.read_csv(DRU_LC)
        plt.plot(d_df["fraction"] * 100, d_df["f1"], marker='o', linestyle='-', label="Proposed DRU-VQC (4 points)")
        
    plt.title("Learning Curve Comparison")
    plt.xlabel("Percentage of Training Data (%)")
    plt.ylabel("F1 Score")
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(LEARNING_CURVE_PNG, dpi=300)
    print(f"Saved learning curve plot to {LEARNING_CURVE_PNG}")

def print_ablation_paragraph():
    text = (
        "\n--- Ablation Paragraph for Report ---\n"
        "The ablation pipeline shares the variational training approach with the proposed model "
        "but retains PCA compression and IQPEmbedding encoding. The accuracy delta between the ablation "
        "and the proposed model therefore reflects the encoding contribution. While it is not a perfect isolation "
        "since both the encoding method and feature selection method change simultaneously, the training setup "
        "is held constant, and the delta is strongly indicative of the encoding's impact.\n"
        "-------------------------------------\n"
    )
    print(text)

def main():
    draw_circuit()
    build_comparison_table()
    plot_learning_curve()
    print_ablation_paragraph()

if __name__ == "__main__":
    main()
