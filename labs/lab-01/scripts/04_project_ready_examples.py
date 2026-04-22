from pathlib import Path

import numpy as np
from scipy.io import wavfile


LAB_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = LAB_DIR / "assets" / "Games.wav"
MAX_SECONDS = 20
WINDOW_SECONDS = 5


def load_audio_excerpt():
    # Lecture du fichier audio.
    sr, y = wavfile.read(AUDIO_PATH)

    # Conversion en mono si stéréo.
    if y.ndim == 2:
        y = y.mean(axis=1)

    # Normalisation pour travailler en flottants.
    y = y.astype(np.float32) / 32768.0

    # On garde un extrait court pour le lab.
    y = y[: min(len(y), int(sr * MAX_SECONDS))]
    return sr, y


def feature_vector(segment, sr):
    # ZCR
    zcr = np.mean(np.abs(np.diff(np.sign(segment))) > 0)

    # Spectre
    freqs = np.fft.rfftfreq(len(segment), d=1 / sr)
    spec = np.abs(np.fft.rfft(segment))
    spec_sum = spec.sum()

    # Features fréquentielles
    centroid = (freqs * spec).sum() / spec_sum
    bandwidth = np.sqrt(((freqs - centroid) ** 2 * spec).sum() / spec_sum)

    # RMS (énergie moyenne)
    rms = np.sqrt(np.mean(segment**2))

    return np.array([zcr, centroid, bandwidth, rms], dtype=np.float64)


def build_feature_matrix(y, sr):
    window_size = int(sr * WINDOW_SECONDS)
    vectors = []

    for i in range(0, len(y) - window_size + 1, window_size):
        segment = y[i : i + window_size]
        vectors.append(feature_vector(segment, sr))

    return np.vstack(vectors)


def main():
    sr, y = load_audio_excerpt()
    matrix = build_feature_matrix(y, sr)

    print("=== Mini dataset de features ===")
    print("segment_id, zcr, centroid_hz, bandwidth_hz, rms")
    for idx, row in enumerate(matrix):
        print(
            idx,
            round(float(row[0]), 6),
            round(float(row[1]), 2),
            round(float(row[2]), 2),
            round(float(row[3]), 6),
        )

    # Comparaison de profils sonores.
    if len(matrix) >= 2:
        dist_0_1 = np.linalg.norm(matrix[0] - matrix[1])
        print("\n=== Comparaison de profils ===")
        print("distance(segment_0, segment_1) =", round(float(dist_0_1), 4))

    # Ranking de recommandations.
    ref_idx = 0
    distances = np.linalg.norm(matrix - matrix[ref_idx], axis=1)
    ranking = np.argsort(distances)

    print("\n=== Ranking de similarité (ref = segment_0) ===")
    for idx in ranking:
        print(f"segment_{idx} -> distance={distances[idx]:.4f}")


if __name__ == "__main__":
    main()
