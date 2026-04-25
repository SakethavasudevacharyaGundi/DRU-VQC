import os
import json
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

SEED = 42

BASE_DIR = r"c:\Users\saket\OneDrive\Documents\Quantum"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TRAIN_PCA_CSV = os.path.join(OUTPUT_DIR, "pca_train_4d.csv")
TEST_PCA_CSV = os.path.join(OUTPUT_DIR, "pca_test_4d.csv")

METRICS_JSON = os.path.join(OUTPUT_DIR, "classical_baselines_metrics.json")
METRICS_CSV = os.path.join(OUTPUT_DIR, "classical_baselines_metrics.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "classical_baselines_overfit_summary.json")


def load_pca_split(path):
    df = pd.read_csv(path)

    required_cols = {"pc1", "pc2", "pc3", "pc4", "binary_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    X = df[["pc1", "pc2", "pc3", "pc4"]].copy()
    y = df["binary_label"].astype(int).copy()
    return X, y


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = {
        "accuracy": accuracy_score(y_train, train_pred),
        "f1": f1_score(y_train, train_pred, zero_division=0),
        "precision": precision_score(y_train, train_pred, zero_division=0),
        "recall": recall_score(y_train, train_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_train, train_pred).tolist(),
    }

    test_metrics = {
        "accuracy": accuracy_score(y_test, test_pred),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }

    return train_metrics, test_metrics


def metric_gaps(train_metrics, test_metrics):
    return {
        "accuracy_gap": train_metrics["accuracy"] - test_metrics["accuracy"],
        "f1_gap": train_metrics["f1"] - test_metrics["f1"],
        "precision_gap": train_metrics["precision"] - test_metrics["precision"],
        "recall_gap": train_metrics["recall"] - test_metrics["recall"],
    }


def main():
    if not os.path.exists(TRAIN_PCA_CSV):
        raise FileNotFoundError(f"Training PCA file not found: {TRAIN_PCA_CSV}")
    if not os.path.exists(TEST_PCA_CSV):
        raise FileNotFoundError(f"Test PCA file not found: {TEST_PCA_CSV}")

    X_train, y_train = load_pca_split(TRAIN_PCA_CSV)
    X_test, y_test = load_pca_split(TEST_PCA_CSV)

    svc_rbf = SVC(kernel="rbf", random_state=SEED)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
    )

    svc_train_metrics, svc_test_metrics = evaluate_model(
        svc_rbf, X_train, y_train, X_test, y_test
    )
    rf_train_metrics, rf_test_metrics = evaluate_model(
        rf, X_train, y_train, X_test, y_test
    )

    results = {
        "SVC_RBF": {
            "train": svc_train_metrics,
            "test": svc_test_metrics,
            "gaps": metric_gaps(svc_train_metrics, svc_test_metrics),
        },
        "Random_Forest": {
            "train": rf_train_metrics,
            "test": rf_test_metrics,
            "gaps": metric_gaps(rf_train_metrics, rf_test_metrics),
        },
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "SVC_RBF",
                "split": "train",
                "accuracy": svc_train_metrics["accuracy"],
                "f1": svc_train_metrics["f1"],
                "precision": svc_train_metrics["precision"],
                "recall": svc_train_metrics["recall"],
            },
            {
                "model": "SVC_RBF",
                "split": "test",
                "accuracy": svc_test_metrics["accuracy"],
                "f1": svc_test_metrics["f1"],
                "precision": svc_test_metrics["precision"],
                "recall": svc_test_metrics["recall"],
            },
            {
                "model": "Random_Forest",
                "split": "train",
                "accuracy": rf_train_metrics["accuracy"],
                "f1": rf_train_metrics["f1"],
                "precision": rf_train_metrics["precision"],
                "recall": rf_train_metrics["recall"],
            },
            {
                "model": "Random_Forest",
                "split": "test",
                "accuracy": rf_test_metrics["accuracy"],
                "f1": rf_test_metrics["f1"],
                "precision": rf_test_metrics["precision"],
                "recall": rf_test_metrics["recall"],
            },
        ]
    )
    metrics_df.to_csv(METRICS_CSV, index=False)

    summary = {
        "SVC_RBF": {
            "train_metrics": svc_train_metrics,
            "test_metrics": svc_test_metrics,
            "gaps": metric_gaps(svc_train_metrics, svc_test_metrics),
            "overfit_flag": bool(metric_gaps(svc_train_metrics, svc_test_metrics)["f1_gap"] > 0.05),
        },
        "Random_Forest": {
            "train_metrics": rf_train_metrics,
            "test_metrics": rf_test_metrics,
            "gaps": metric_gaps(rf_train_metrics, rf_test_metrics),
            "overfit_flag": bool(metric_gaps(rf_train_metrics, rf_test_metrics)["f1_gap"] > 0.05),
        },
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Step 6 verification complete.")
    print("\nTrain/Test metrics:")
    print(metrics_df.to_string(index=False))

    print("\nOverfitting summary:")
    for model_name, payload in summary.items():
        gaps = payload["gaps"]
        print(
            f"{model_name}: "
            f"F1 gap={gaps['f1_gap']:.6f}, "
            f"Accuracy gap={gaps['accuracy_gap']:.6f}, "
            f"Overfit flag={payload['overfit_flag']}"
        )

    print(f"\nSaved: {METRICS_JSON}")
    print(f"Saved: {METRICS_CSV}")
    print(f"Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()