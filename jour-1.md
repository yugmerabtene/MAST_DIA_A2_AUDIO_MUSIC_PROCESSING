# Jour 1

## Comprendre la structure musicale et les signaux audio (3H30)

**Notions clés**
- **Signal audio** : suite de valeurs qui représente le son dans le temps.
- **Fréquence d'échantillonnage** : nombre de mesures du signal par seconde.
- **Forme d'onde** : représentation de l'amplitude au cours du temps.
- **Spectrogramme** : représentation temps-fréquence de l'énergie sonore.
- **Feature** : caractère numérique extrait du signal pour décrire le son de façon compacte.
- **Feature engineering** : choix et calcul des features utiles pour la suite du projet.

**Introduction**
Cette première partie pose les bases de la musique et du signal audio pour comprendre ce que l'on manipule ensuite en traitement audio.
L'objectif est de passer d'une écoute intuitive à une lecture technique du son.

**Explication**
On relie les notions musicales simples à leur traduction dans un signal numérique : hauteur, rythme, fréquence, amplitude et durée.
Une guitare, par exemple, permet d'illustrer concrètement la différence entre une note jouée, sa fréquence fondamentale et sa forme d'onde enregistrée.
Dans le cours, Librosa est l'outil de référence pour visualiser ces signaux ; ici, le code utilise SciPy pour rester exécutable dans le dépôt.

**Contexte**
En analyse musicale, il faut savoir lire un extrait sonore avant de pouvoir en extraire des caractéristiques exploitables.
C'est la base pour préparer des données audio avant toute classification ou recommandation.
Le lab associé à ce chapitre utilise l'extrait `labs/lab-01/assets/exemple_cours.wav`.

**Tracer la forme d'onde**
La forme d'onde est la première lecture du signal. Elle permet de voir les variations d'amplitude et les grandes structures temporelles avant de passer au spectrogramme.

**Formule mathematique**

$$
f_s = \frac{N}{T}
$$

**Lecture de la formule**
"f indice s égale N sur T."

**Sens de la formule**
La fréquence d'échantillonnage relie le nombre d'échantillons à la durée observée.

**Lien avec la théorie**
Plus la fréquence d'échantillonnage est élevée, plus le signal est détaillé dans le temps.

**Décomposition mathématique**
- `f_s` : fréquence d'échantillonnage en hertz
- `N` : nombre total d'échantillons
- `T` : durée du signal en secondes

**Résultat attendu**
Savoir faire le lien entre notions musicales et représentation numérique d'un son.
Savoir expliquer ce que représente un signal audio et pourquoi sa structure temporelle et fréquentielle compte.

**Code**

```python
# Import des bibliotheques pour le calcul numerique et l'affichage.
import numpy as np
# Signal permet de calculer le spectrogramme.
from scipy import signal
# Lecture du fichier audio au format WAV.
from scipy.io import wavfile
# Matplotlib sert a tracer la forme d'onde et le spectrogramme.
import matplotlib.pyplot as plt

# Chargement de l'extrait audio du lab.
sr, y = wavfile.read("labs/lab-01/assets/exemple_cours.wav")
# Conversion en flottants entre -1 et 1.
y = y.astype(float) / 32768.0
# Duree du signal en secondes.
duration = len(y) / sr
# Axe temporel associe a chaque echantillon.
t = np.linspace(0, duration, len(y), endpoint=False)

# Affichage des infos de base.
print("Fréquence d'échantillonnage:", sr)
print("Durée (s):", duration)

# Trace de la forme d'onde.
plt.figure(figsize=(10, 3))
plt.plot(t, y)
plt.title("Forme d'onde")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.show()

# Calcul et affichage du spectrogramme.
plt.figure(figsize=(10, 3))
f, tt, Zxx = signal.stft(y, fs=sr, nperseg=1024)
plt.pcolormesh(tt, f, np.abs(Zxx), shading='gouraud')
plt.colorbar(label='Amplitude')
plt.title("Spectrogramme")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.show()
```

**Explication du code**
Ce bloc charge un extrait audio, affiche sa forme d'onde, puis calcule un spectrogramme pour visualiser l'information temps-frequence. L'objectif est de relier la notion de signal audio brut à une lecture plus technique du son.

## Extraire les caractéristiques audio (3h30)

**Introduction**
Cette partie montre comment transformer un signal audio en descripteurs numériques utilisables pour l'analyse et la classification.
On cherche ici à représenter un morceau par quelques mesures robustes plutôt que par l'onde brute.

**Explication**
On extrait des mesures simples comme le zero crossing rate ou le centre spectral pour décrire le contenu sonore.
Ces descripteurs résument l'énergie, la brillance ou l'activité du signal sous une forme compacte.

**Pourquoi ces features ?**
Les features réduisent un signal complexe à quelques variables interprétables. Elles rendent possible la comparaison entre morceaux et l'entraînement d'un modèle.

**Utiliser Librosa pour extraire des features fréquentiels et harmoniques**
La logique de ce chapitre consiste à résumer le signal avec des descripteurs comme le zero crossing rate, le centroid spectral, la bandwidth et la structure harmonique.

**Contexte**
Ces caractéristiques servent à comparer des morceaux ou à préparer un dataset pour un modèle de machine learning.
Dans un système musical, elles peuvent aider à distinguer des genres, des instruments ou des ambiances.

**Comparer les profils sonores de différents genres musicaux**
En observant les features et les spectrogrammes de plusieurs morceaux, on peut repérer des signatures sonores différentes selon les genres. Cette comparaison prépare directement le terrain pour la classification.

**Éléments techniques importants**
- **ZCR** : donne une idée de l'agitation du signal.
- **Centroid spectral** : mesure où se concentre l'énergie dans les fréquences.
- **Bandwidth spectrale** : mesure l'étendue de cette énergie.
- **Structure harmonique** : décrit les fréquences liées à la fondamentale et à ses harmoniques.
- **STFT** : découpe le signal en petites fenêtres pour observer son évolution dans le temps.
- **FFT** : transforme le signal du temps vers les fréquences.

**Analyser les spectrogrammes**
Le spectrogramme aide à voir comment les harmoniques évoluent dans le temps et à comparer plus facilement deux genres musicaux.

**Formule mathematique**

$$
ZCR = \frac{1}{N-1} \sum_{n=1}^{N-1} \mathbf{1}(x_n x_{n-1} < 0)
$$

**Lecture de la formule**
"Z C R égale un sur N moins 1 fois la somme de l'indicatrice de x n fois x n moins 1 inférieur à zéro."

**Sens de la formule**
Le zero crossing rate mesure combien de fois le signal change de signe.

**Lien avec la théorie**
Un ZCR élevé correspond souvent à un signal plus agité ou plus bruité, tandis qu'un ZCR faible correspond à un signal plus stable.

**Interprétation audio**
Un son percussif ou bruité tend à avoir un ZCR plus fort qu'un son pur et stable comme une note tenue.

**Décomposition mathématique**
- `N` : nombre total d'échantillons pris en compte
- `x_n` : valeur de l'échantillon numéro `n`
- `1[x_n x_{n-1} < 0]` : vaut `1` si le signal change de signe entre deux échantillons consécutifs, sinon `0`

**Résultat attendu**
Savoir extraire et interpréter des features audio de base.
Savoir expliquer à quoi servent ces features dans une chaîne d'analyse musicale.
Savoir relier les nombres calculés à ce qu'on entend dans le morceau.

**Code**

```python
# Import des outils de calcul numerique.
import numpy as np
# Chargement du fichier audio de reference.
from scipy.io import wavfile

# Lecture de l'extrait audio.
sr, y = wavfile.read("labs/lab-01/assets/exemple_cours.wav")
# Normalisation du signal pour travailler en flottants.
y = y.astype(float) / 32768.0

# Zero crossing rate : nombre de changements de signe.
zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)

# Transformation vers le domaine frequentiel.
freqs = np.fft.rfftfreq(len(y), d=1 / sr)
spec = np.abs(np.fft.rfft(y))
spec_sum = spec.sum()
# Centroid spectral : frequence moyenne ponderee par l'energie.
centroid = (freqs * spec).sum() / spec_sum
# Largeur spectrale autour du centroid.
bandwidth = np.sqrt(((freqs - centroid) ** 2 * spec).sum() / spec_sum)

# Affichage des descripteurs calcules.
print("ZCR:", zcr)
print("Spectral centroid:", centroid)
print("Spectral bandwidth:", bandwidth)
```

**Explication du code**
Ce bloc transforme le signal audio en mesures compactes. Le ZCR donne une idee de l'agitation du signal, tandis que le centroid et la bandwidth décrivent la repartition de l'energie dans les frequences.

## Synthèse du jour

- Lire un signal audio comme un objet numérique.
- Relier notions musicales et représentation temps-fréquence.
- Extraire des features simples à partir d'un fichier audio de référence.
