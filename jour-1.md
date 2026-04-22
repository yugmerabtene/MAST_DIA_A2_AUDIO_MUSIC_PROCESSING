# Jour 1

## 1. Comprendre la structure musicale et les signaux audio

**Introduction**
Cette premiere partie pose les bases de la musique et du signal audio pour comprendre ce que l'on manipule ensuite en traitement audio.
L'objectif est de passer d'une ecoute intuitive a une lecture technique du son.

**Explication**
On relie les notions musicales simples a leur traduction dans un signal numerique : hauteur, rythme, frequence, amplitude et duree.
Une guitare, par exemple, permet d'illustrer concretement la difference entre une note jouee, sa frequence fondamentale et sa forme d'onde enregistrée.

**Contexte**
En analyse musicale, il faut savoir lire un extrait sonore avant de pouvoir en extraire des caracteristiques exploitables.
C'est la base pour preparer des donnees audio avant toute classification ou recommandation.

**Formule mathematique**

$$
f_s = \frac{N}{T}
$$

**Lecture de la formule**
"f indice s egale N sur T."

**Sens de la formule**
La frequence d'echantillonnage relie le nombre d'echantillons a la duree observee.

**Decomposition mathematique**
- `f_s` : frequence d'echantillonnage
- `N` : nombre d'echantillons
- `T` : duree du signal

**Resultat attendu**
Savoir faire le lien entre notions musicales et representation numerique d'un son.
Savoir expliquer ce que represente un signal audio et pourquoi sa structure temporelle et frequentielle compte.

**Code**

```python
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = "sample.wav"
y, sr = librosa.load(audio_path, sr=None)

print("Frequence d'echantillonnage:", sr)
print("Duree (s):", librosa.get_duration(y=y, sr=sr))

plt.figure(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr)
plt.title("Forme d'onde")
plt.show()

plt.figure(figsize=(10, 3))
D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogramme")
plt.show()
```

## 2. Extraire les caracteristiques audio

**Introduction**
Cette partie montre comment transformer un signal audio en descripteurs numeriques utilisables pour l'analyse et la classification.
On cherche ici a representer un morceau par quelques mesures robustes plutot que par l'onde brute.

**Explication**
On extrait des mesures simples comme le zero crossing rate ou le centre spectral pour decrire le contenu sonore.
Ces descripteurs resumeraient l'energie, la brillance ou l'activite du signal sous une forme compacte.

**Contexte**
Ces caracteristiques servent a comparer des morceaux ou a preparer un dataset pour un modele de machine learning.
Dans un systeme musical, elles peuvent aider a distinguer des genres, des instruments ou des ambiances.

**Formule mathematique**

$$
ZCR = \frac{1}{N-1} \sum_{n=1}^{N-1} \mathbf{1}(x_n x_{n-1} < 0)
$$

**Lecture de la formule**
"Z C R egale un sur N moins 1 fois la somme de l'indicatrice de x n fois x n moins 1 inferieur a zero."

**Sens de la formule**
Le zero crossing rate mesure combien de fois le signal change de signe.

**Decomposition mathematique**
- `N` : nombre d'echantillons
- `x_n` : echantillon numero n
- `\mathbf{1}(...)` : fonction indicatrice

**Resultat attendu**
Savoir extraire et interpreter des features audio de base.
Savoir expliquer a quoi servent ces features dans une chaine d'analyse musicale.

**Code**

```python
import librosa

audio_path = "sample.wav"
y, sr = librosa.load(audio_path, sr=None)

zcr = librosa.feature.zero_crossing_rate(y)
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

print("ZCR:", zcr.mean())
print("Spectral centroid:", centroid.mean())
print("Spectral bandwidth:", bandwidth.mean())
print("Tempo:", tempo)
```

## Synthese du jour

- Lire un signal audio comme un objet numerique.
- Relier notions musicales et representation temps-frequence.
- Extraire des features simples avec Librosa.
