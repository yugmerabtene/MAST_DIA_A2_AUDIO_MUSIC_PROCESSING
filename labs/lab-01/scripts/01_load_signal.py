from pathlib import Path

import numpy as np
from scipy.io import wavfile


LAB_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = LAB_DIR / "assets" / "Games.wav"
MAX_SECONDS = 20


def main():
    # Lecture du fichier audio brut.
    sr, y = wavfile.read(AUDIO_PATH)

    # Nombre de canaux avant conversion.
    channels = 1 if y.ndim == 1 else y.shape[1]

    # Passage en mono si le fichier est stéréo.
    if y.ndim == 2:
        y = y.mean(axis=1)

    # Normalisation entre -1 et 1.
    y = y.astype(np.float32) / 32768.0

    # Limitation à un extrait court pour le lab.
    max_samples = min(len(y), int(sr * MAX_SECONDS))
    y = y[:max_samples]
    duration = len(y) / sr

    print(f"audio_path = {AUDIO_PATH}")
    print(f"sample_rate = {sr}")
    print(f"channels_before_mono = {channels}")
    print(f"analyzed_shape = {y.shape}")
    print(f"analyzed_duration_seconds = {duration:.2f}")


if __name__ == "__main__":
    main()
