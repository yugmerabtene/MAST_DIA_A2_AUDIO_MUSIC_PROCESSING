# Lab 01 - Signaux audio et features

## Objectif

Manipuler un extrait audio simple, visualiser sa forme d'onde et son spectrogramme, puis extraire quelques features de base.

## Vocabulaire

- **Signal audio** : suite de valeurs qui represente le son dans le temps.
- **Feature** : mesure numerique compacte extraite du signal.
- **Forme d'onde** : lecture temporelle de l'amplitude.
- **Spectrogramme** : lecture temps-frequence de l'energie.
- **ZCR** : nombre de changements de signe du signal.
- **Centroid spectral** : frequence moyenne ponderee par l'energie.
- **Bandwidth spectrale** : largeur de dispersion autour du centroid.
- **STFT** : methode pour observer l'evolution des frequences dans le temps.

## Fichier audio

L'extrait de reference est dans `assets/exemple_cours.wav`.

## Etape 1 - Charger le signal

**Resultat attendu**
Lire le fichier audio et verifier sa frequence d'echantillonnage.

**Lien avec la theorie**
On passe de la notion abstraite de signal a une representation numerique concrete.

```python
from scipy.io import wavfile

sr, y = wavfile.read("assets/exemple_cours.wav")
print("sr =", sr)
print("shape =", y.shape)
```

## Etape 2 - Visualiser la forme d'onde

**Resultat attendu**
Observer la variation d'amplitude dans le temps.

**Lien avec la theorie**
La forme d'onde illustre directement l'energie du signal au cours du temps.

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

**Resultat attendu**
Voir comment l'energie se repartit selon le temps et les frequences.

**Lien avec la theorie**
Le spectrogramme relie la partie temporelle a la partie frequentielle du signal.

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

**Resultat attendu**
Calculer quelques descripteurs simples pour caracteriser le son.

**Lien avec la theorie**
Ces features condensent l'information utile pour comparer deux sons ou preparer un modele.

**Pourquoi ces features ?**
Le but est de transformer le son brut en variables exploitables pour comparer, classer ou recommander.

**Ce qu'il faut comprendre avant de coder**
- Le signal brut est trop long et trop riche pour etre utilise tel quel directement dans un modele simple.
- Une feature est un resume du signal qui garde une information utile.
- Plus les features sont pertinentes, plus le modele aura de chances d'etre efficace.

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

Le lab montre comment partir d'un fichier audio simple pour aller vers des representations exploitables en traitement audio.
