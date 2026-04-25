import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

SEED = 42

BASE_DIR = r"c:\Users\saket\OneDrive\Documents\Quantum"
INPUT_TRAIN_CSV = os.path.join(BASE_DIR, "outputs", "step3_train_800.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Rebuild labels defensively if needed
    if "binary_label" not in df.columns:
        if "source_class" in df.columns:
            df["binary_label"] = (df["source_class"] != "benign").astype(int)
        elif "label" in df.columns:
            df["binary_label"] = (~df["label"].astype(str).eq("BenignTraffic")).astype(int)
        else:
            raise ValueError("No label column found to build binary_label.")

    if "source_class" not in df.columns:
        if "label" in df.columns:
            df["source_class"] = np.where(
                df["label"].astype(str).eq("BenignTraffic"),
                "benign",
                "attack",
            )
        else:
            df["source_class"] = np.where(df["binary_label"] == 0, "benign", "attack")

    return df


def get_feature_matrix(df: pd.DataFrame):
    drop_cols = [c for c in ["binary_label", "source_class", "label"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["binary_label"].astype(int)

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).copy()

    # Replace inf with NaN and drop columns with NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    bad_cols = [c for c in X.columns if X[c].isnull().any()]
    if bad_cols:
        X = X.drop(columns=bad_cols)

    return X, y


def rank_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    mi = mutual_info_classif(X, y, random_state=SEED)
    ranked = pd.DataFrame(
        {
            "feature": X.columns,
            "mi_score": mi,
        }
    ).sort_values(
        by=["mi_score", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    ranked["rank"] = np.arange(1, len(ranked) + 1)

    def batch_for_rank(r):
        if 1 <= r <= 4:
            return "batch_1"
        if 5 <= r <= 8:
            return "batch_2"
        if 9 <= r <= 12:
            return "batch_3"
        return "other"

    ranked["batch"] = ranked["rank"].apply(batch_for_rank)
    return ranked


def main():
    if not os.path.exists(INPUT_TRAIN_CSV):
        raise FileNotFoundError(
            f"Training CSV not found: {INPUT_TRAIN_CSV}"
        )

    df = load_training_data(INPUT_TRAIN_CSV)
    X, y = get_feature_matrix(df)

    if X.shape[1] == 0:
        raise ValueError("No numeric features available after cleaning.")

    ranked = rank_features(X, y)

    top12 = ranked.head(12).copy()

    ranked_csv = os.path.join(OUTPUT_DIR, "mi_ranking_all_features.csv")
    top12_csv = os.path.join(OUTPUT_DIR, "mi_top12_batches.csv")
    top12_json = os.path.join(OUTPUT_DIR, "mi_top12_batches.json")

    ranked.to_csv(ranked_csv, index=False)
    top12.to_csv(top12_csv, index=False)

    with open(top12_json, "w", encoding="utf-8") as f:
        json.dump(top12.to_dict(orient="records"), f, indent=2)

    print("Training rows:", len(df))
    print("Feature count:", X.shape[1])
    print("Saved:", ranked_csv)
    print("Saved:", top12_csv)
    print("Saved:", top12_json)
    print("\nTop 12 features:")
    print(top12[["feature", "mi_score", "rank", "batch"]].to_string(index=False))


if __name__ == "__main__":
    main()