import numpy as np
from scipy.io import wavfile
from pathlib import Path


def main():
    # Parametres du signal de reference.
    sr = 22050
    duration = 4.0
    # Axe temporel du signal.
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Segments harmoniques simples pour simuler une petite progression musicale.
    segments = [
        (0.0, 1.0, [220.0, 261.63, 329.63]),
        (1.0, 2.0, [261.63, 329.63, 392.00]),
        (2.0, 3.0, [196.00, 246.94, 392.00]),
        (3.0, 4.0, [164.81, 207.65, 329.63]),
    ]

    # Construction du signal morceau par morceau.
    y = np.zeros_like(t)
    for start, end, freqs in segments:
        # Fenetre temporelle du segment courant.
        idx = (t >= start) & (t < end)
        tt = t[idx] - start
        # Somme de plusieurs sinusoides pour generer un accord simple.
        sig = sum(np.sin(2 * np.pi * f * tt) for f in freqs) / len(freqs)
        # Enveloppe douce pour eviter les clics aux frontieres.
        env = np.sin(np.pi * np.clip(tt, 0, 1)) ** 2
        y[idx] = sig * env

    # Normalisation avant export en 16 bits.
    y = 0.9 * y / np.max(np.abs(y))
    path = Path(__file__).parent / "assets" / "exemple_cours.wav"
    # Ecriture du fichier WAV de reference.
    wavfile.write(path, sr, (y * 32767).astype(np.int16))
    print(f"written: {path}")


if __name__ == "__main__":
    main()
