# Jour 1

## 1. Comprendre la structure musicale et les signaux audio

**Introduction**
Cette premiere partie pose les bases de la musique et du signal audio pour comprendre ce que l'on manipule ensuite en traitement audio.

**Explication**
On relie les notions musicales simples a leur traduction dans un signal numerique : hauteur, rythme, frequence, amplitude et duree.

**Contexte**
En analyse musicale, il faut savoir lire un extrait sonore avant de pouvoir en extraire des caracteristiques exploitables.

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
L'etudiant sait faire le lien entre notions musicales et representation numerique d'un son.

**Code**

```python
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
```

## 2. Extraire les caracteristiques audio

**Introduction**
Cette partie montre comment transformer un signal audio en descripteurs numeriques utilisables pour l'analyse et la classification.

**Explication**
On extrait des mesures simples comme le zero crossing rate ou le centre spectral pour decrire le contenu sonore.

**Contexte**
Ces caracteristiques servent a comparer des morceaux ou a preparer un dataset pour un modele de machine learning.

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
L'etudiant sait extraire et interpreter des features audio de base.

**Code**

```python
import librosa

audio_path = "sample.wav"
y, sr = librosa.load(audio_path, sr=None)

zcr = librosa.feature.zero_crossing_rate(y)
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

print("ZCR:", zcr.mean())
print("Spectral centroid:", centroid.mean())
print("Spectral bandwidth:", bandwidth.mean())
```

## Synthese du jour

- Lire un signal audio comme un objet numerique.
- Relier notions musicales et representation temps-frequence.
- Extraire des features simples avec Librosa.
