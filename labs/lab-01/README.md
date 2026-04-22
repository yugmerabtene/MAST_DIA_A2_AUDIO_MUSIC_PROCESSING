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
# Lecture du fichier audio de reference.
from scipy.io import wavfile

sr, y = wavfile.read("assets/exemple_cours.wav")
print("sr =", sr)
print("shape =", y.shape)
```

**Explication du code**
On charge le fichier audio et on verifie sa forme interne. C'est la première etape avant toute analyse du signal.

## Etape 2 - Visualiser la forme d'onde

**Résultat attendu**
Observer la variation d'amplitude dans le temps.

**Lien avec la théorie**
La forme d'onde illustre directement l'énergie du signal au cours du temps.

```python
# Outil de tracé.
import matplotlib.pyplot as plt
# Lecture du fichier audio.
from scipy.io import wavfile

# Chargement du signal.
sr, y = wavfile.read("assets/exemple_cours.wav")
# Normalisation pour un affichage lisible.
y = y.astype(float) / 32768.0
# Axe temporel en secondes.
t = [i / sr for i in range(len(y))]

# Trace de la forme d'onde.
plt.plot(t, y)
plt.title("Forme d'onde")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.show()
```

**Explication du code**
Cette étape visualise l'amplitude du signal dans le temps. Elle permet de voir la structure globale du morceau avant de passer au domaine frequentiel.

## Etape 3 - Visualiser le spectrogramme

**Résultat attendu**
Voir comment l'énergie se répartit selon le temps et les fréquences.

**Lien avec la théorie**
Le spectrogramme relie la partie temporelle à la partie fréquentielle du signal.

```python
# Calcul numerique.
import numpy as np
# Outil de tracé.
import matplotlib.pyplot as plt
# Calcul du spectrogramme.
from scipy import signal
# Lecture du fichier audio.
from scipy.io import wavfile

# Chargement et normalisation du signal.
sr, y = wavfile.read("assets/exemple_cours.wav")
y = y.astype(float) / 32768.0
# STFT pour observer le signal par fenetres.
f, tt, Zxx = signal.stft(y, fs=sr, nperseg=1024)

# Affichage du spectrogramme.
plt.pcolormesh(tt, f, np.abs(Zxx), shading="gouraud")
plt.title("Spectrogramme")
plt.xlabel("Temps (s)")
plt.ylabel("Frequence (Hz)")
plt.colorbar(label="Amplitude")
plt.show()
```

**Explication du code**
Le spectrogramme montre comment l'energie se repartit entre le temps et les frequences. C'est utile pour repérer les changements de timbre, d'intensité ou de contenu harmonique.

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
# Calcul numerique.
import numpy as np
# Lecture du fichier audio.
from scipy.io import wavfile

# Chargement et normalisation du signal.
sr, y = wavfile.read("assets/exemple_cours.wav")
y = y.astype(float) / 32768.0

# Zero crossing rate : nombre de changements de signe.
zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)
# Axe frequentiel.
freqs = np.fft.rfftfreq(len(y), d=1 / sr)
# Spectre en amplitude.
spec = np.abs(np.fft.rfft(y))
spec_sum = spec.sum()
# Centre spectral.
centroid = (freqs * spec).sum() / spec_sum

# Largeur spectrale autour du centre.
bandwidth = np.sqrt(((freqs - centroid) ** 2 * spec).sum() / spec_sum)

print("ZCR =", zcr)
print("Centroid =", centroid)
print("Bandwidth =", bandwidth)
```

**Explication du code**
Ce bloc extrait des features simples à partir du signal. Elles compressent l'information audio dans quelques variables utiles pour comparer des morceaux ou alimenter un modele.

## Conclusion

Le lab montre comment partir d'un fichier audio simple pour aller vers des représentations exploitables en traitement audio.
