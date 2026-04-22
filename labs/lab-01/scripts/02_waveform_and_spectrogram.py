from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


LAB_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = LAB_DIR / "assets" / "Games.wav"
OUTPUT_DIR = LAB_DIR / "outputs"
MAX_SECONDS = 20


def load_audio_excerpt():
    # Lecture du fichier audio.
    sr, y = wavfile.read(AUDIO_PATH)

    # Conversion en mono si besoin.
    if y.ndim == 2:
        y = y.mean(axis=1)

    # Normalisation entre -1 et 1.
    y = y.astype(np.float32) / 32768.0

    # Conservation d'un extrait court pour garder des figures lisibles.
    max_samples = min(len(y), int(sr * MAX_SECONDS))
    y = y[:max_samples]
    t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
    return sr, t, y


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sr, t, y = load_audio_excerpt()

    # Trace de la forme d'onde.
    plt.figure(figsize=(12, 4))
    plt.plot(t, y)
    plt.title("Forme d'onde - Games.wav")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "waveform.png", dpi=150)
    plt.close()

    # Calcul et affichage du spectrogramme.
    plt.figure(figsize=(12, 4))
    f, tt, zxx = signal.stft(y, fs=sr, nperseg=2048)
    plt.pcolormesh(tt, f, np.abs(zxx), shading="gouraud")
    plt.title("Spectrogramme - Games.wav")
    plt.xlabel("Temps (s)")
    plt.ylabel("Fréquence (Hz)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spectrogram.png", dpi=150)
    plt.close()

    print(f"waveform_saved = {OUTPUT_DIR / 'waveform.png'}")
    print(f"spectrogram_saved = {OUTPUT_DIR / 'spectrogram.png'}")


if __name__ == "__main__":
    main()
