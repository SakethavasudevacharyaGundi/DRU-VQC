import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

import pennylane as qml


SEED = 42
WIRES = 4

BASE_DIR = r"c:\Users\saket\OneDrive\Documents\Quantum"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TRAIN_PCA_CSV = os.path.join(OUTPUT_DIR, "pca_train_4d.csv")
TEST_PCA_CSV = os.path.join(OUTPUT_DIR, "pca_test_4d.csv")

TRAIN_KERNEL_CACHE = os.path.join(OUTPUT_DIR, "qsvm_train_kernel.npy")
TEST_KERNEL_CACHE = os.path.join(OUTPUT_DIR, "qsvm_test_kernel.npy")
CACHE_META_PATH = os.path.join(OUTPUT_DIR, "qsvm_kernel_cache_meta.json")

METRICS_JSON = os.path.join(OUTPUT_DIR, "qsvm_metrics.json")
METRICS_CSV = os.path.join(OUTPUT_DIR, "qsvm_metrics.csv")
LEARNING_CURVE_CSV = os.path.join(OUTPUT_DIR, "qsvm_learning_curve.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "qsvm_summary.json")


def file_digest(path: str) -> str:
	hasher = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			hasher.update(chunk)
	return hasher.hexdigest()


def load_pca_split(path: str) -> tuple[np.ndarray, np.ndarray]:
	df = pd.read_csv(path)
	required_cols = ["pc1", "pc2", "pc3", "pc4", "binary_label"]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in {path}: {missing}")

	X = df[["pc1", "pc2", "pc3", "pc4"]].to_numpy(dtype=float)
	y = df["binary_label"].astype(int).to_numpy()
	return X, y


def build_kernel(device: qml.Device):
	wires = list(range(WIRES))

	@qml.qnode(device)
	def fidelity_circuit(x1, x2):
		qml.IQPEmbedding(x1, wires=wires)
		qml.adjoint(qml.IQPEmbedding)(x2, wires=wires)
		return qml.probs(wires=wires)

	return lambda x1, x2: float(fidelity_circuit(x1, x2)[0])


def cache_is_valid(train_path: str, test_path: str) -> bool:
	if not (os.path.exists(TRAIN_KERNEL_CACHE) and os.path.exists(TEST_KERNEL_CACHE) and os.path.exists(CACHE_META_PATH)):
		return False

	with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
		meta = json.load(f)

	expected = {
		"seed": SEED,
		"train_digest": file_digest(train_path),
		"test_digest": file_digest(test_path),
		"wires": WIRES,
		"embedding": "IQPEmbedding",
	}
	for key, value in expected.items():
		if meta.get(key) != value:
			return False
	return True


def save_cache_meta(train_path: str, test_path: str, train_shape, test_shape, train_elapsed: float, test_elapsed: float):
	meta = {
		"seed": SEED,
		"wires": WIRES,
		"embedding": "IQPEmbedding",
		"train_digest": file_digest(train_path),
		"test_digest": file_digest(test_path),
		"train_shape": list(train_shape),
		"test_shape": list(test_shape),
		"train_kernel_cache": TRAIN_KERNEL_CACHE,
		"test_kernel_cache": TEST_KERNEL_CACHE,
		"train_kernel_seconds": train_elapsed,
		"test_kernel_seconds": test_elapsed,
	}
	with open(CACHE_META_PATH, "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)


def compute_or_load_kernels(X_train: np.ndarray, X_test: np.ndarray, train_path: str, test_path: str, use_cache: bool):
	device = qml.device("lightning.qubit", wires=WIRES)
	kernel = build_kernel(device)

	if use_cache and cache_is_valid(train_path, test_path):
		train_kernel = np.load(TRAIN_KERNEL_CACHE)
		test_kernel = np.load(TEST_KERNEL_CACHE)
		return train_kernel, test_kernel, {"train_seconds": None, "test_seconds": None, "cached": True}

	start = time.perf_counter()
	train_kernel = qml.kernels.square_kernel_matrix(
		X_train,
		kernel,
		assume_normalized_kernel=True,
	)
	train_seconds = time.perf_counter() - start

	start = time.perf_counter()
	test_kernel = qml.kernels.kernel_matrix(X_test, X_train, kernel)
	test_seconds = time.perf_counter() - start

	if use_cache:
		np.save(TRAIN_KERNEL_CACHE, train_kernel)
		np.save(TEST_KERNEL_CACHE, test_kernel)
		save_cache_meta(train_path, test_path, X_train.shape, X_test.shape, train_seconds, test_seconds)

	return train_kernel, test_kernel, {"train_seconds": train_seconds, "test_seconds": test_seconds, "cached": False}


def evaluate_precomputed_svc(K_train, y_train, K_test, y_test):
	model = SVC(kernel="precomputed", random_state=SEED)
	model.fit(K_train, y_train)

	train_pred = model.predict(K_train)
	test_pred = model.predict(K_test)

	train_metrics = {
		"accuracy": accuracy_score(y_train, train_pred),
		"f1": f1_score(y_train, train_pred, zero_division=0),
		"precision": precision_score(y_train, train_pred, zero_division=0),
		"recall": recall_score(y_train, train_pred, zero_division=0),
	}
	test_metrics = {
		"accuracy": accuracy_score(y_test, test_pred),
		"f1": f1_score(y_test, test_pred, zero_division=0),
		"precision": precision_score(y_test, test_pred, zero_division=0),
		"recall": recall_score(y_test, test_pred, zero_division=0),
	}
	return model, train_pred, test_pred, train_metrics, test_metrics


def learning_curve_points(K_train_full, K_test_full, y_train, y_test, fractions):
	curve_rows = []
	for fraction in fractions:
		if fraction == 1.0:
			subset_idx = np.arange(len(y_train))
		else:
			subset_size = max(2, int(round(len(y_train) * fraction)))
			subset_splitter = StratifiedShuffleSplit(
				n_splits=1,
				train_size=subset_size,
				random_state=SEED,
			)
			subset_idx, _ = next(subset_splitter.split(K_train_full, y_train))
			subset_idx = np.array(sorted(subset_idx))

		K_subset_train = K_train_full[np.ix_(subset_idx, subset_idx)]
		K_subset_test = K_test_full[:, subset_idx]
		y_subset = y_train[subset_idx]

		model = SVC(kernel="precomputed", random_state=SEED)
		model.fit(K_subset_train, y_subset)
		test_pred = model.predict(K_subset_test)
		f1 = f1_score(y_test, test_pred, zero_division=0)

		curve_rows.append(
			{
				"fraction": float(fraction),
				"train_size": int(len(subset_idx)),
				"f1": float(f1),
			}
		)

	return curve_rows


def stratified_limit(X: np.ndarray, y: np.ndarray, limit: int | None):
	if limit is None or limit >= len(y):
		return X, y, np.arange(len(y))

	splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=SEED)
	subset_idx, _ = next(splitter.split(X, y))
	subset_idx = np.array(sorted(subset_idx))
	return X[subset_idx], y[subset_idx], subset_idx


def main():
	parser = argparse.ArgumentParser(description="QSVM benchmark on 4D PCA features.")
	parser.add_argument(
		"--train-limit",
		type=int,
		default=None,
		help="Optional stratified cap on the training set for smoke testing.",
	)
	parser.add_argument(
		"--force-recompute",
		action="store_true",
		help="Ignore cached kernels and recompute them.",
	)
	args = parser.parse_args()

	os.makedirs(OUTPUT_DIR, exist_ok=True)

	if not os.path.exists(TRAIN_PCA_CSV):
		raise FileNotFoundError(f"Training PCA file not found: {TRAIN_PCA_CSV}")
	if not os.path.exists(TEST_PCA_CSV):
		raise FileNotFoundError(f"Test PCA file not found: {TEST_PCA_CSV}")

	X_train, y_train = load_pca_split(TRAIN_PCA_CSV)
	X_test, y_test = load_pca_split(TEST_PCA_CSV)

	X_train, y_train, train_subset_idx = stratified_limit(X_train, y_train, args.train_limit)

	use_cache = args.train_limit is None
	if args.force_recompute and use_cache:
		for path in [TRAIN_KERNEL_CACHE, TEST_KERNEL_CACHE, CACHE_META_PATH]:
			if os.path.exists(path):
				os.remove(path)

	train_kernel, test_kernel, kernel_info = compute_or_load_kernels(
		X_train,
		X_test,
		TRAIN_PCA_CSV,
		TEST_PCA_CSV,
		use_cache=use_cache,
	)

	_, _, _, train_metrics, test_metrics = evaluate_precomputed_svc(
		train_kernel,
		y_train,
		test_kernel,
		y_test,
	)

	curve_rows = learning_curve_points(
		train_kernel,
		test_kernel,
		y_train,
		y_test,
		fractions=[0.25, 1.0],
	)

	metrics_rows = [
		{"model": "QSVM", "split": "train", **train_metrics},
		{"model": "QSVM", "split": "test", **test_metrics},
	]
	metrics_df = pd.DataFrame(metrics_rows)
	metrics_df.to_csv(METRICS_CSV, index=False)

	with open(METRICS_JSON, "w", encoding="utf-8") as f:
		json.dump(
			{
				"train": train_metrics,
				"test": test_metrics,
				"kernel_info": kernel_info,
				"train_samples": int(len(y_train)),
				"test_samples": int(len(y_test)),
			},
			f,
			indent=2,
		)

	learning_curve_df = pd.DataFrame(curve_rows)
	learning_curve_df.to_csv(LEARNING_CURVE_CSV, index=False)

	summary = {
		"seed": SEED,
		"wires": WIRES,
		"embedding": "IQPEmbedding",
		"train_shape": list(X_train.shape),
		"test_shape": list(X_test.shape),
		"cached": kernel_info["cached"],
		"kernel_cache_train": TRAIN_KERNEL_CACHE,
		"kernel_cache_test": TEST_KERNEL_CACHE,
		"metrics_csv": METRICS_CSV,
		"metrics_json": METRICS_JSON,
		"learning_curve_csv": LEARNING_CURVE_CSV,
		"test_metrics": test_metrics,
		"train_metrics": train_metrics,
		"train_subset_indices_count": int(len(train_subset_idx)),
	}
	with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print("QSVM run complete.")
	print("\nMetrics:")
	print(metrics_df.to_string(index=False))
	print("\nLearning curve:")
	print(learning_curve_df.to_string(index=False))
	print(f"\nSaved: {METRICS_JSON}")
	print(f"Saved: {METRICS_CSV}")
	print(f"Saved: {LEARNING_CURVE_CSV}")
	print(f"Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
	main()
