# Lab 01 - Signaux audio et features

## Objectif

Manipuler un extrait audio simple, visualiser sa forme d'onde et son spectrogramme, puis extraire quelques features de base.

## Vocabulaire

- **Signal audio** : suite de valeurs qui représente le son dans le temps.
- **Feature** : mesure numérique compacte extraite du signal.
- **Forme d'onde** : lecture temporelle de l'amplitude.
- **Spectrogramme** : lecture temps-fréquence de l'énergie.
- **ZCR** : nombre de changements de signe du signal.
- **Centroid spectral** : fréquence moyenne pondérée par l'énergie.
- **Bandwidth spectrale** : largeur de dispersion autour du centroid.
- **STFT** : méthode pour observer l'évolution des fréquences dans le temps.

## Fichier audio

L'extrait de référence est dans `assets/exemple_cours.wav`.

## Etape 1 - Charger le signal

**Résultat attendu**
Lire le fichier audio et vérifier sa fréquence d'échantillonnage.

**Lien avec la théorie**
On passe de la notion abstraite de signal à une représentation numérique concrète.

```python
from scipy.io import wavfile

sr, y = wavfile.read("assets/exemple_cours.wav")
print("sr =", sr)
print("shape =", y.shape)
```

## Etape 2 - Visualiser la forme d'onde

**Résultat attendu**
Observer la variation d'amplitude dans le temps.

**Lien avec la théorie**
La forme d'onde illustre directement l'énergie du signal au cours du temps.

```python
import matplotlib.pyplot as plt
from scipy.io import wavfile

sr, y = wavfile.read("assets/exemple_cours.wav")
y = y.astype(float) / 32768.0
t = [i / sr for i in range(len(y))]

plt.plot(t, y)
plt.title("Forme d'onde")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.show()
```

## Etape 3 - Visualiser le spectrogramme

**Résultat attendu**
Voir comment l'énergie se répartit selon le temps et les fréquences.

**Lien avec la théorie**
Le spectrogramme relie la partie temporelle à la partie fréquentielle du signal.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sr, y = wavfile.read("assets/exemple_cours.wav")
y = y.astype(float) / 32768.0
f, tt, Zxx = signal.stft(y, fs=sr, nperseg=1024)

plt.pcolormesh(tt, f, np.abs(Zxx), shading="gouraud")
plt.title("Spectrogramme")
plt.xlabel("Temps (s)")
plt.ylabel("Frequence (Hz)")
plt.colorbar(label="Amplitude")
plt.show()
```

## Etape 4 - Extraire des features

**Résultat attendu**
Calculer quelques descripteurs simples pour caractériser le son.

**Lien avec la théorie**
Ces features condensent l'information utile pour comparer deux sons ou préparer un modèle.

**Pourquoi ces features ?**
Le but est de transformer le son brut en variables exploitables pour comparer, classer ou recommander.

**Ce qu'il faut comprendre avant de coder**
- Le signal brut est trop long et trop riche pour être utilisé tel quel directement dans un modèle simple.
- Une feature est un résumé du signal qui garde une information utile.
- Plus les features sont pertinentes, plus le modèle aura de chances d'être efficace.

```python
import numpy as np
from scipy.io import wavfile

sr, y = wavfile.read("assets/exemple_cours.wav")
y = y.astype(float) / 32768.0

zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)
freqs = np.fft.rfftfreq(len(y), d=1 / sr)
spec = np.abs(np.fft.rfft(y))
spec_sum = spec.sum()
centroid = (freqs * spec).sum() / spec_sum

print("ZCR =", zcr)
print("Centroid =", centroid)
```

## Conclusion

Le lab montre comment partir d'un fichier audio simple pour aller vers des représentations exploitables en traitement audio.
