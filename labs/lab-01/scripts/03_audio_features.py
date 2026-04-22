from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks


LAB_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = LAB_DIR / "assets" / "Games.wav"
MAX_SECONDS = 20


def load_audio_excerpt():
    # Lecture du fichier audio.
    sr, y = wavfile.read(AUDIO_PATH)

    # Conversion en mono si le fichier est stéréo.
    if y.ndim == 2:
        y = y.mean(axis=1)

    # Normalisation entre -1 et 1.
    y = y.astype(np.float32) / 32768.0

    # Conservation d'un extrait court pour accélérer les calculs.
    max_samples = min(len(y), int(sr * MAX_SECONDS))
    y = y[:max_samples]
    return sr, y


def main():
    sr, y = load_audio_excerpt()

    # Zero crossing rate : nombre moyen de changements de signe.
    zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)

    # Passage dans le domaine fréquentiel.
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)
    spectrum = np.abs(np.fft.rfft(y))
    spectrum_sum = spectrum.sum()

    # Centre spectral : fréquence moyenne pondérée par l'énergie.
    centroid = (freqs * spectrum).sum() / spectrum_sum

    # Bandwidth : étendue de l'énergie autour du centre spectral.
    bandwidth = np.sqrt(((freqs - centroid) ** 2 * spectrum).sum() / spectrum_sum)

    # Recherche de quelques fréquences dominantes.
    peaks, _ = find_peaks(spectrum, distance=50)
    peak_strength = spectrum[peaks]
    top_peaks = peaks[np.argsort(peak_strength)[-5:]]
    dominant_freqs = sorted(freqs[top_peaks])

    print(f"zcr = {zcr:.6f}")
    print(f"spectral_centroid_hz = {centroid:.2f}")
    print(f"spectral_bandwidth_hz = {bandwidth:.2f}")
    print("dominant_frequencies_hz =", [round(freq, 2) for freq in dominant_freqs])


if __name__ == "__main__":
    main()
