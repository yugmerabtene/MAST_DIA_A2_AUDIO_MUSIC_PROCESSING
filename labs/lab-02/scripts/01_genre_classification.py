from pathlib import Path

import numpy as np


LAB_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = LAB_DIR / "assets" / "mini_genre_dataset.csv"
FEATURE_COLUMNS = ["zcr", "centroid_hz", "bandwidth_hz", "rms"]


def load_dataset():
    raw = np.genfromtxt(DATA_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")
    X = np.column_stack([raw[col].astype(np.float64) for col in FEATURE_COLUMNS])
    y = raw["genre"].astype(str)
    return X, y


def stratified_split(y, test_size=0.33, seed=42):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        idx = rng.permutation(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_indices.extend(idx[:n_test])
        train_indices.extend(idx[n_test:])

    return np.array(train_indices), np.array(test_indices)


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def knn_predict(train_x, train_y, test_x, k=3):
    preds = []
    for sample in test_x:
        distances = np.linalg.norm(train_x - sample, axis=1)
        nn = np.argsort(distances)[:k]
        labels, counts = np.unique(train_y[nn], return_counts=True)
        preds.append(labels[np.argmax(counts)])
    return np.array(preds)


def build_confusion(y_true, y_pred, classes):
    label_to_idx = {label: i for i, label in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        cm[label_to_idx[truth], label_to_idx[pred]] += 1
    return cm


def print_report(y_true, y_pred, classes):
    cm = build_confusion(y_true, y_pred, classes)
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total else 0.0

    print("\nclassification_report:")
    print("label          precision  recall  f1-score  support")

    precisions = []
    recalls = []
    f1s = []

    for i, label in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"{label:<14} {precision:>9.3f} {recall:>7.3f} {f1:>9.3f} {support:>8d}")

    print(f"\naccuracy: {accuracy:.4f}")
    print(
        "macro avg      "
        f"{np.mean(precisions):>9.3f} {np.mean(recalls):>7.3f} {np.mean(f1s):>9.3f} {total:>8d}"
    )


def main():
    X, y = load_dataset()
    classes = np.unique(y)

    train_idx, test_idx = stratified_split(y, test_size=0.33, seed=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_scaled, X_test_scaled = standardize(X_train, X_test)
    y_pred = knn_predict(X_train_scaled, y_train, X_test_scaled, k=3)
    cm = build_confusion(y_test, y_pred, classes)

    print("=== Classification de genres (lab-02) ===")
    print("samples:", len(X), "| train:", len(X_train), "| test:", len(X_test))
    print("\nconfusion_matrix:")
    print(cm)
    print_report(y_test, y_pred, classes)


if __name__ == "__main__":
    main()
