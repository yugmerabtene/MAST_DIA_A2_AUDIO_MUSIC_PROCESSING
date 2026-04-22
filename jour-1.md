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
Le lab associe a ce chapitre utilise l'extrait `labs/lab-01/assets/exemple_cours.wav`.

**Formule mathematique**

$$
f_s = \frac{N}{T}
$$

**Lecture de la formule**
"f indice s egale N sur T."

**Sens de la formule**
La frequence d'echantillonnage relie le nombre d'echantillons a la duree observee.

**Lien avec la theorie**
Plus la frequence d'echantillonnage est elevee, plus le signal est detaille dans le temps.

**Decomposition mathematique**
- `f_s` : frequence d'echantillonnage en hertz
- `N` : nombre total d'echantillons
- `T` : duree du signal en secondes

**Resultat attendu**
Savoir faire le lien entre notions musicales et representation numerique d'un son.
Savoir expliquer ce que represente un signal audio et pourquoi sa structure temporelle et frequentielle compte.

**Code**

```python
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

sr, y = wavfile.read("labs/lab-01/assets/exemple_cours.wav")
y = y.astype(float) / 32768.0
duration = len(y) / sr
t = np.linspace(0, duration, len(y), endpoint=False)

print("Frequence d'echantillonnage:", sr)
print("Duree (s):", duration)

plt.figure(figsize=(10, 3))
plt.plot(t, y)
plt.title("Forme d'onde")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.show()

plt.figure(figsize=(10, 3))
f, tt, Zxx = signal.stft(y, fs=sr, nperseg=1024)
plt.pcolormesh(tt, f, np.abs(Zxx), shading='gouraud')
plt.colorbar(label='Amplitude')
plt.title("Spectrogramme")
plt.xlabel("Temps (s)")
plt.ylabel("Frequence (Hz)")
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

**Lien avec la theorie**
Un ZCR eleve correspond souvent a un signal plus agite ou plus bruité, tandis qu'un ZCR faible correspond a un signal plus stable.

**Decomposition mathematique**
- `N` : nombre total d'echantillons pris en compte
- `x_n` : valeur de l'echantillon numero `n`
- `1(x_n x_{n-1} < 0)` : vaut `1` si le signal change de signe entre deux echantillons consecutifs, sinon `0`

**Resultat attendu**
Savoir extraire et interpreter des features audio de base.
Savoir expliquer a quoi servent ces features dans une chaine d'analyse musicale.

**Code**

```python
import numpy as np
from scipy.io import wavfile

sr, y = wavfile.read("labs/lab-01/assets/exemple_cours.wav")
y = y.astype(float) / 32768.0

zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)

freqs = np.fft.rfftfreq(len(y), d=1 / sr)
spec = np.abs(np.fft.rfft(y))
spec_sum = spec.sum()
centroid = (freqs * spec).sum() / spec_sum
bandwidth = np.sqrt(((freqs - centroid) ** 2 * spec).sum() / spec_sum)

print("ZCR:", zcr)
print("Spectral centroid:", centroid)
print("Spectral bandwidth:", bandwidth)
```

## Synthese du jour

- Lire un signal audio comme un objet numerique.
- Relier notions musicales et representation temps-frequence.
- Extraire des features simples a partir d'un fichier audio de reference.
