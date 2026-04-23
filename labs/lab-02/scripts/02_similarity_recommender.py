from pathlib import Path

import numpy as np


LAB_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = LAB_DIR / "assets" / "mini_genre_dataset.csv"
FEATURE_COLUMNS = ["zcr", "centroid_hz", "bandwidth_hz", "rms"]
TARGET_TRACK_ID = "track_09"
TOP_K = 5


def load_dataset():
    raw = np.genfromtxt(DATA_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")
    track_ids = raw["track_id"].astype(str)
    genres = raw["genre"].astype(str)
    X = np.column_stack([raw[col].astype(np.float64) for col in FEATURE_COLUMNS])
    return track_ids, genres, X


def main():
    track_ids, genres, X = load_dataset()

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    X_scaled = (X - mean) / std

    target_idx = np.where(track_ids == TARGET_TRACK_ID)[0][0]
    distances = np.linalg.norm(X_scaled - X_scaled[target_idx], axis=1)
    ranking = np.argsort(distances)

    print("=== Recommandation par similarite (lab-02) ===")
    print("target:", TARGET_TRACK_ID)
    print("\nTop recommandations:")

    count = 0
    for idx in ranking:
        if idx == target_idx:
            continue
        print(
            f"{track_ids[idx]} | genre={genres[idx]} | distance={distances[idx]:.4f}"
        )
        count += 1
        if count >= TOP_K:
            break


if __name__ == "__main__":
    main()
