import os
import glob
import json
import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)

DATASET_DIR = r"c:\Users\saket\OneDrive\Documents\Quantum\dataset"
OUTPUT_DIR = r"c:\Users\saket\OneDrive\Documents\Quantum\outputs"

# Fast-path buffer collection per class (enough for exact Step 3 sample)
NEED_PER_CLASS = {
    "benign": 500,
    "ddos": 500,
    "mirai": 500,
}

def collect_fast_path_rows(files, chunksize=100000):
    b_parts, d_parts, m_parts = [], [], []
    counts = {"benign": 0, "ddos": 0, "mirai": 0}
    rows_scanned = 0

    for fp in files:
        if all(counts[k] >= NEED_PER_CLASS[k] for k in NEED_PER_CLASS):
            break

        for chunk in pd.read_csv(fp, chunksize=chunksize):
            rows_scanned += len(chunk)
            labels = chunk["label"].astype(str)

            if counts["benign"] < NEED_PER_CLASS["benign"]:
                b = chunk[labels == "BenignTraffic"]
                if not b.empty:
                    take = min(len(b), NEED_PER_CLASS["benign"] - counts["benign"])
                    b_parts.append(b.iloc[:take].copy())
                    counts["benign"] += take

            if counts["ddos"] < NEED_PER_CLASS["ddos"]:
                d = chunk[labels.str.startswith("DDoS-")]
                if not d.empty:
                    take = min(len(d), NEED_PER_CLASS["ddos"] - counts["ddos"])
                    d_parts.append(d.iloc[:take].copy())
                    counts["ddos"] += take

            if counts["mirai"] < NEED_PER_CLASS["mirai"]:
                m = chunk[labels.str.startswith("Mirai-")]
                if not m.empty:
                    take = min(len(m), NEED_PER_CLASS["mirai"] - counts["mirai"])
                    m_parts.append(m.iloc[:take].copy())
                    counts["mirai"] += take

            if all(counts[k] >= NEED_PER_CLASS[k] for k in NEED_PER_CLASS):
                break

    if any(counts[k] < NEED_PER_CLASS[k] for k in NEED_PER_CLASS):
        raise RuntimeError(f"Insufficient rows collected: {counts}")

    benign_df = pd.concat(b_parts, ignore_index=True)
    ddos_df = pd.concat(d_parts, ignore_index=True)
    mirai_df = pd.concat(m_parts, ignore_index=True)
    return benign_df, ddos_df, mirai_df, counts, rows_scanned

def step2_clean_merge(benign_df, ddos_df, mirai_df):
    benign_df = benign_df.copy()
    ddos_df = ddos_df.copy()
    mirai_df = mirai_df.copy()

    benign_df["source_class"] = "benign"
    ddos_df["source_class"] = "ddos"
    mirai_df["source_class"] = "mirai"

    merged = pd.concat([benign_df, ddos_df, mirai_df], ignore_index=True)

    # Binary label: 0 benign, 1 attack
    merged["binary_label"] = (merged["source_class"] != "benign").astype(int)

    # Replace inf with NaN and drop invalid columns
    merged = merged.replace([np.inf, -np.inf], np.nan)

    keep_non_numeric = {"label", "source_class", "binary_label"}
    non_numeric_cols = [
        c for c in merged.columns
        if c not in keep_non_numeric and not pd.api.types.is_numeric_dtype(merged[c])
    ]
    null_cols = [c for c in merged.columns if merged[c].isnull().any()]

    drop_cols = sorted(set(non_numeric_cols + null_cols))
    clean = merged.drop(columns=drop_cols)

    # Drop original multiclass label after class extraction
    if "label" in clean.columns:
        clean = clean.drop(columns=["label"])

    feature_cols = [c for c in clean.columns if c not in ["source_class", "binary_label"]]
    return merged, clean, drop_cols, feature_cols

def step3_sample_split(clean):
    # Exact sampling counts
    s_b = clean[clean["source_class"] == "benign"].sample(n=333, random_state=SEED)
    s_d = clean[clean["source_class"] == "ddos"].sample(n=333, random_state=SEED)
    s_m = clean[clean["source_class"] == "mirai"].sample(n=334, random_state=SEED)
    sampled = pd.concat([s_b, s_d, s_m], ignore_index=True)

    # Exact 80/20 split: 800 train, 200 test
    # Class-stratified allocation preserving near-proportions
    # benign=66, ddos=67, mirai=67 => test total 200
    alloc = {"benign": 66, "ddos": 67, "mirai": 67}

    idx = np.arange(len(sampled))
    cls = sampled["source_class"].values

    test_idx = []
    for c in ["benign", "ddos", "mirai"]:
        c_idx = idx[cls == c]
        c_idx = RNG.permutation(c_idx)
        test_idx.extend(c_idx[:alloc[c]].tolist())

    test_idx = np.array(RNG.permutation(test_idx))
    mask = np.ones(len(sampled), dtype=bool)
    mask[test_idx] = False
    train_idx = np.where(mask)[0]

    train_df = sampled.iloc[train_idx].reset_index(drop=True)
    test_df = sampled.iloc[test_idx].reset_index(drop=True)

    # Person-2 mandatory indices (against sampled row order)
    test_indices_df = pd.DataFrame({"sampled_index": test_idx.tolist()})
    return sampled, train_df, test_df, test_indices_df, test_idx.tolist()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
    if not files:
        raise RuntimeError(f"No CSV files found in {DATASET_DIR}")

    benign_df, ddos_df, mirai_df, counts, rows_scanned = collect_fast_path_rows(files)
    merged, clean, drop_cols, feature_cols = step2_clean_merge(benign_df, ddos_df, mirai_df)
    sampled, train_df, test_df, test_indices_df, test_indices = step3_sample_split(clean)

    # Save outputs
    merged.to_csv(os.path.join(OUTPUT_DIR, "step2_merged_raw_1500.csv"), index=False)
    clean.to_csv(os.path.join(OUTPUT_DIR, "step2_clean_1500.csv"), index=False)
    sampled.to_csv(os.path.join(OUTPUT_DIR, "step3_sampled_1000.csv"), index=False)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "step3_train_800.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "step3_test_200.csv"), index=False)
    test_indices_df.to_csv(os.path.join(OUTPUT_DIR, "step3_test_indices.csv"), index=False)

    summary = {
        "seed": SEED,
        "rows_scanned_fast_path": int(rows_scanned),
        "collected_counts": counts,
        "step2_merged_shape": list(merged.shape),
        "step2_clean_shape": list(clean.shape),
        "step2_dropped_columns_count": len(drop_cols),
        "step2_numeric_feature_count": len(feature_cols),
        "step2_binary_distribution": {str(k): int(v) for k, v in clean["binary_label"].value_counts().to_dict().items()},
        "step3_sampled_shape": list(sampled.shape),
        "step3_sampled_class_counts": {str(k): int(v) for k, v in sampled["source_class"].value_counts().to_dict().items()},
        "step3_train_shape": list(train_df.shape),
        "step3_test_shape": list(test_df.shape),
        "step3_train_class_counts": {str(k): int(v) for k, v in train_df["source_class"].value_counts().to_dict().items()},
        "step3_test_class_counts": {str(k): int(v) for k, v in test_df["source_class"].value_counts().to_dict().items()},
        "step3_train_binary_counts": {str(k): int(v) for k, v in train_df["binary_label"].value_counts().to_dict().items()},
        "step3_test_binary_counts": {str(k): int(v) for k, v in test_df["binary_label"].value_counts().to_dict().items()},
        "step3_test_index_count": len(test_indices),
        "step3_test_index_preview_30": test_indices[:30],
        "output_dir": OUTPUT_DIR,
    }

    with open(os.path.join(OUTPUT_DIR, "step2_step3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()