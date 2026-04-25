import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42

BASE_DIR = r"c:\Users\saket\OneDrive\Documents\Quantum"
INPUT_TRAIN_CSV = os.path.join(BASE_DIR, "outputs", "step3_train_800.csv")
INPUT_TEST_CSV = os.path.join(BASE_DIR, "outputs", "step3_test_200.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binary_label" not in df.columns:
        if "source_class" in df.columns:
            df["binary_label"] = (df["source_class"] != "benign").astype(int)
        else:
            raise ValueError(f"binary_label not found and cannot be reconstructed in {path}")

    if "source_class" not in df.columns:
        df["source_class"] = np.where(df["binary_label"] == 0, "benign", "attack")

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["binary_label", "source_class", "label"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Keep numeric features only
    X = X.select_dtypes(include=[np.number]).copy()

    # Remove inf and columns with any nulls
    X = X.replace([np.inf, -np.inf], np.nan)
    bad_cols = [c for c in X.columns if X[c].isnull().any()]
    if bad_cols:
        X = X.drop(columns=bad_cols)

    return X


def main():
    if not os.path.exists(INPUT_TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV not found: {INPUT_TRAIN_CSV}")
    if not os.path.exists(INPUT_TEST_CSV):
        raise FileNotFoundError(f"Test CSV not found: {INPUT_TEST_CSV}")

    train_df = load_split(INPUT_TRAIN_CSV)
    test_df = load_split(INPUT_TEST_CSV)

    X_train = get_feature_matrix(train_df)
    X_test = get_feature_matrix(test_df)

    # Ensure identical feature columns in same order
    common_features = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_features].copy()
    X_test = X_test[common_features].copy()

    if len(common_features) == 0:
        raise ValueError("No common numeric features found between train and test splits.")

    # Fit on training only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=4, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    train_pca_df = pd.DataFrame(
        X_train_pca,
        columns=["pc1", "pc2", "pc3", "pc4"]
    )
    train_pca_df["binary_label"] = train_df["binary_label"].astype(int).values
    train_pca_df["source_class"] = train_df["source_class"].values

    test_pca_df = pd.DataFrame(
        X_test_pca,
        columns=["pc1", "pc2", "pc3", "pc4"]
    )
    test_pca_df["binary_label"] = test_df["binary_label"].astype(int).values
    test_pca_df["source_class"] = test_df["source_class"].values

    # Save exports
    train_csv = os.path.join(OUTPUT_DIR, "pca_train_4d.csv")
    test_csv = os.path.join(OUTPUT_DIR, "pca_test_4d.csv")
    train_npy = os.path.join(OUTPUT_DIR, "pca_train_4d.npy")
    test_npy = os.path.join(OUTPUT_DIR, "pca_test_4d.npy")
    model_path = os.path.join(OUTPUT_DIR, "pca_scaler_and_model.pkl")
    summary_path = os.path.join(OUTPUT_DIR, "pca_summary.json")

    train_pca_df.to_csv(train_csv, index=False)
    test_pca_df.to_csv(test_csv, index=False)
    np.save(train_npy, X_train_pca)
    np.save(test_npy, X_test_pca)

    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "features": common_features,
            "seed": SEED,
        },
        model_path,
    )

    summary = {
        "seed": SEED,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "pca_train_shape": list(X_train_pca.shape),
        "pca_test_shape": list(X_test_pca.shape),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "feature_count_used": len(common_features),
        "train_csv": train_csv,
        "test_csv": test_csv,
        "train_npy": train_npy,
        "test_npy": test_npy,
        "model_path": model_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nTrain PCA head:")
    print(train_pca_df.head().to_string(index=False))
    print("\nTest PCA head:")
    print(test_pca_df.head().to_string(index=False))


if __name__ == "__main__":
    main()