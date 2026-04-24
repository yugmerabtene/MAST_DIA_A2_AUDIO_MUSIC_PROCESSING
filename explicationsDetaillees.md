**Machine Learning pour l’audio processing**.

L’idée centrale est la suivante :

```text
Fichier audio
↓
Signal numérique
↓
Extraction de features
↓
Features données au modèle
↓
Le modèle apprend avec les labels
↓
Le modèle prédit une classe
```

# 1. Vue globale : feature, label, classe

En apprentissage supervisé, on a généralement trois éléments importants :

| Élément     | Rôle                                    | Exemple audio                           |
| ----------- | --------------------------------------- | --------------------------------------- |
| **Feature** | Ce que le modèle observe                | MFCC, spectrogramme, énergie, fréquence |
| **Label**   | La bonne réponse associée à un fichier  | `chien`, `chat`, `colère`, `rock`       |
| **Classe**  | Une catégorie possible dans le problème | `chien`, `chat`, `voiture`, `pluie`     |

Exemple simple :

```text
audio_001.wav
↓
features : MFCC, énergie, fréquence, spectrogramme
↓
label : chien
↓
classe correcte : chien
```

Donc :

```text
features = données d’entrée
label = réponse correcte pour un exemple
classe = catégorie possible
```

# 2. Qu’est-ce qu’une feature ?

Une **feature** est une caractéristique mesurable extraite d’un fichier audio.

Un modèle de machine learning ne comprend pas directement le son comme un humain. Il a besoin de nombres.

Un fichier audio est donc transformé en données numériques.

```text
Son réel
↓
microphone
↓
fichier audio numérique
↓
signal audio
↓
features numériques
↓
modèle
```

Exemple :

```text
Un humain entend : "aboiement de chien"

Le modèle reçoit plutôt :
- énergie moyenne : 0.74
- fréquence dominante : 850 Hz
- zero crossing rate : 0.12
- MFCC_1 : -210.5
- MFCC_2 : 84.2
- MFCC_3 : -13.7
```

Le modèle ne comprend pas le mot “chien”. Il apprend que certains motifs numériques correspondent souvent à la classe `chien`.

# 3. Le signal audio brut

Avant les features, il faut comprendre ce qu’est un signal audio.

Un son est une vibration. Quand on l’enregistre, cette vibration devient une suite de valeurs numériques.

Exemple simplifié :

```text
audio.wav = [0.01, 0.04, 0.08, 0.03, -0.02, -0.07, -0.05, ...]
```

Chaque nombre représente l’amplitude du signal à un instant précis.

## 3.1 Sample rate

Le **sample rate**, ou fréquence d’échantillonnage, indique combien de mesures sont prises par seconde.

Exemples :

```text
16 000 Hz = 16 000 valeurs par seconde
22 050 Hz = 22 050 valeurs par seconde
44 100 Hz = 44 100 valeurs par seconde
48 000 Hz = 48 000 valeurs par seconde
```

Un fichier audio de 10 secondes à 44 100 Hz contient environ :

```text
44 100 × 10 = 441 000 valeurs numériques
```

Cela fait beaucoup de données brutes. C’est pour cela qu’on extrait des features plus compactes et plus utiles.

## 3.2 Amplitude

L’amplitude représente la force du signal.

```text
amplitude élevée   → son fort
amplitude faible   → son faible
amplitude proche 0 → silence
```

Exemple :

```text
voix calme        → faible amplitude
cri               → forte amplitude
explosion         → très forte amplitude
bruit de fond     → faible amplitude continue
```

L’amplitude seule ne suffit pas pour reconnaître un son, mais elle donne une information importante sur l’intensité.

# 4. Les features temporelles

Les features temporelles sont calculées directement sur le signal audio dans le temps.

Elles regardent la forme de l’onde sonore.

## 4.1 Énergie

L’énergie mesure la puissance globale du son.

Un son fort aura souvent une énergie plus élevée.

```text
son faible → énergie basse
son fort   → énergie élevée
```

Exemples :

```text
chuchotement → énergie faible
cri          → énergie forte
tambour      → pics d’énergie
silence      → énergie proche de zéro
```

Utilisation :

```text
détection de silence
détection d’événements sonores
reconnaissance d’émotion vocale
classification de sons urbains
```

Dans la voix, une émotion comme la colère peut souvent avoir une énergie plus élevée qu’une voix neutre ou triste.

## 4.2 RMS Energy

Le **RMS**, pour Root Mean Square, mesure l’énergie moyenne du signal.

Il permet de mesurer le volume perçu du signal.

Exemple :

```text
audio_001.wav → RMS = 0.03 → son faible
audio_002.wav → RMS = 0.40 → son fort
```

C’est utile pour comparer l’intensité de plusieurs fichiers audio.

## 4.3 Zero Crossing Rate

Le **Zero Crossing Rate**, souvent abrégé **ZCR**, mesure combien de fois le signal traverse la ligne zéro.

Autrement dit, on compte combien de fois le signal passe du positif au négatif ou du négatif au positif.

```text
signal positif → signal négatif = crossing
signal négatif → signal positif = crossing
```

Un son très bruité ou aigu a souvent un ZCR élevé.

Exemples :

```text
son grave        → ZCR plutôt bas
son aigu         → ZCR plus élevé
bruit blanc      → ZCR élevé
voix humaine     → ZCR moyen
percussion douce → ZCR variable
```

Utilisation :

```text
différencier voix et bruit
détecter des consonnes dans la parole
analyser le timbre
classifier des sons courts
```

## 4.4 Durée

La durée est une feature simple mais parfois très utile.

Exemple :

```text
aboiement court       → 0.3 à 1 seconde
sirène                → plusieurs secondes
mot prononcé          → durée variable
note musicale tenue   → durée longue
claquement de porte   → événement très court
```

Dans certains problèmes, la durée aide beaucoup.

Exemple :

```text
un claquement
une alarme
un cri
une phrase
une musique
```

Ces sons n’ont pas forcément les mêmes durées.

## 4.5 Silence et pauses

Dans la voix, les silences sont importants.

On peut extraire :

```text
nombre de pauses
durée moyenne des pauses
durée totale de silence
position des silences
```

Exemple en reconnaissance d’émotion :

```text
voix hésitante → pauses fréquentes
voix triste    → débit lent, pauses longues
voix en colère → pauses courtes, débit rapide
```

# 5. Les features fréquentielles

Les features fréquentielles analysent le contenu du signal en fréquences.

Un son est composé de plusieurs fréquences.

```text
fréquence basse → son grave
fréquence haute → son aigu
```

Exemples :

```text
basse musicale      → fréquences basses
sifflement          → fréquences hautes
voix masculine      → fréquences plutôt basses
voix féminine       → fréquences souvent plus hautes
cymbale             → beaucoup de hautes fréquences
```

## 5.1 Fréquence fondamentale

La fréquence fondamentale correspond à la fréquence principale perçue.

Dans la voix, elle correspond souvent au **pitch**.

```text
pitch bas  → voix grave
pitch haut → voix aiguë
```

Exemple :

```text
voix grave : 100 Hz environ
voix aiguë : 250 Hz environ
sifflement : fréquences beaucoup plus hautes
```

Utilisation :

```text
reconnaissance de locuteur
détection d’émotion
analyse musicale
classification voix homme/femme
détection de mélodie
```

## 5.2 Pitch

Le **pitch** est la hauteur perçue du son.

Attention : le pitch est lié à la fréquence, mais il correspond plutôt à la perception humaine.

Exemple :

```text
pitch élevé   → voix aiguë, cri, excitation
pitch faible  → voix grave, calme, tristesse possible
pitch variable → expressivité, chant, émotion
```

En audio émotionnel :

```text
joie    → pitch souvent plus élevé
colère  → pitch élevé et énergie forte
tristesse → pitch souvent plus bas
neutre → pitch plus stable
```

## 5.3 Spectral Centroid

Le **spectral centroid** indique où se trouve le “centre de gravité” du spectre audio.

Il donne une idée de la brillance du son.

```text
spectral centroid bas  → son sombre, grave
spectral centroid haut → son brillant, aigu
```

Exemples :

```text
basse électrique → centroid bas
cymbale          → centroid haut
voix grave       → centroid plutôt bas
sifflement       → centroid haut
```

Utilisation :

```text
classification musicale
reconnaissance d’instruments
analyse du timbre
classification de sons environnementaux
```

## 5.4 Spectral Bandwidth

Le **spectral bandwidth** mesure l’étalement des fréquences autour du centre spectral.

Un son peut être concentré sur peu de fréquences ou étalé sur beaucoup de fréquences.

```text
bandwidth faible → son pur ou simple
bandwidth élevé  → son riche, bruité ou complexe
```

Exemples :

```text
diapason      → bandwidth faible
bruit blanc   → bandwidth élevé
cymbale       → bandwidth élevé
voix humaine  → bandwidth moyen
```

## 5.5 Spectral Rolloff

Le **spectral rolloff** indique la fréquence en dessous de laquelle se trouve une grande partie de l’énergie du signal.

Par exemple, on peut dire :

```text
85 % de l’énergie du signal se trouve sous telle fréquence
```

Utilisation :

```text
détection de sons brillants
classification musicale
différenciation parole/musique/bruit
```

## 5.6 Spectral Contrast

Le **spectral contrast** mesure la différence entre les pics et les creux du spectre.

Il est utile pour analyser le timbre.

Exemple :

```text
instrument harmonique clair → contraste spectral marqué
bruit diffus                → contraste spectral plus faible
```

Utilisation :

```text
classification de genres musicaux
reconnaissance d’instruments
analyse de texture sonore
```

# 6. Le spectrogramme

Le **spectrogramme** est l’une des représentations les plus importantes en audio processing.

Il représente le son selon trois dimensions :

```text
axe horizontal   → temps
axe vertical     → fréquence
intensité/couleur → énergie
```

On peut le voir comme une image du son.

```text
audio brut
↓
transformation temps-fréquence
↓
spectrogramme
↓
modèle de deep learning
```

## 6.1 Pourquoi le spectrogramme est important ?

Le signal audio brut est difficile à interpréter directement.

Le spectrogramme permet de voir :

```text
quelles fréquences sont présentes
à quel moment elles apparaissent
combien de temps elles durent
avec quelle intensité elles sont présentes
```

Exemple :

```text
un aboiement → formes courtes et intenses
une sirène   → courbe fréquentielle régulière
la pluie     → texture diffuse
la parole    → motifs liés aux syllabes
la musique   → structures rythmiques et harmoniques
```

## 6.2 Spectrogramme et deep learning

En deep learning, un spectrogramme peut être traité comme une image.

```text
audio.wav
↓
spectrogramme
↓
CNN
↓
classe prédite
```

Un **CNN**, réseau de neurones convolutif, peut apprendre à reconnaître des motifs visuels dans le spectrogramme.

Exemple :

```text
motifs de chien
motifs de pluie
motifs de voiture
motifs de voix
```

# 7. Mel Spectrogram

Le **Mel Spectrogram** est une version du spectrogramme adaptée à la perception humaine.

L’oreille humaine ne perçoit pas les fréquences de manière linéaire.

La différence entre 100 Hz et 200 Hz est très perceptible.

Mais la différence entre 10 000 Hz et 10 100 Hz est beaucoup moins perceptible.

L’échelle Mel corrige cela.

```text
spectrogramme classique
↓
projection sur l’échelle Mel
↓
Mel Spectrogram
```

Utilisation :

```text
reconnaissance vocale
classification audio
détection d’événements sonores
deep learning audio
```

Le Mel Spectrogram est souvent meilleur que le spectrogramme brut pour les tâches liées à la perception humaine.

# 8. MFCC

Les **MFCC**, pour **Mel Frequency Cepstral Coefficients**, sont des features très utilisées en traitement audio.

Elles résument la forme du spectre sonore en tenant compte de la perception humaine.

```text
audio brut
↓
découpage en petites fenêtres
↓
transformation fréquentielle
↓
échelle Mel
↓
logarithme
↓
compression
↓
coefficients MFCC
```

## 8.1 À quoi servent les MFCC ?

Les MFCC permettent de représenter le timbre et la forme globale du son.

Ils sont très utilisés pour :

```text
reconnaissance vocale
classification de sons
identification de locuteur
détection d’émotion
classification musicale
```

## 8.2 Exemple concret

Pour un fichier audio, on peut obtenir :

```text
MFCC_1  = -210.4
MFCC_2  = 78.2
MFCC_3  = 12.9
MFCC_4  = -34.1
...
MFCC_13 = 5.7
```

Chaque coefficient décrit une partie de la forme sonore.

Le modèle apprend par exemple :

```text
certains profils de MFCC → voix
certains profils de MFCC → musique
certains profils de MFCC → bruit mécanique
certains profils de MFCC → animal
```

## 8.3 MFCC moyens

Comme un fichier audio contient plusieurs instants, on peut calculer les MFCC sur plusieurs fenêtres.

Ensuite, on peut prendre :

```text
moyenne des MFCC
écart-type des MFCC
minimum des MFCC
maximum des MFCC
```

Exemple :

```text
audio_001.wav
↓
MFCC sur chaque frame
↓
moyenne de chaque coefficient
↓
vecteur final de features
```

# 9. Chroma Features

Les **chroma features** sont utilisées surtout pour la musique.

Elles représentent les notes musicales indépendamment de leur octave.

Il existe 12 classes de notes :

```text
Do
Do#
Ré
Ré#
Mi
Fa
Fa#
Sol
Sol#
La
La#
Si
```

Les chroma features indiquent quelles notes sont présentes dans le signal.

Utilisation :

```text
reconnaissance d’accords
analyse harmonique
classification de genres musicaux
détection de tonalité
recommandation musicale
```

Exemple :

```text
musique avec beaucoup de Do, Mi, Sol
↓
probablement accord de Do majeur
```

# 10. Tempo et rythme

Pour la musique, le **tempo** indique la vitesse du morceau.

Il est souvent mesuré en BPM.

```text
BPM = beats per minute
```

Exemple :

```text
60 BPM  → lent
120 BPM → moyen/dansant
180 BPM → rapide
```

Utilisation :

```text
classification musicale
détection d’ambiance
recommandation
analyse de danse
segmentation musicale
```

Exemple :

```text
musique classique lente → tempo faible
techno                  → tempo régulier et élevé
rap                     → rythme marqué
jazz                    → rythme plus complexe
```

# 11. Les features en deep learning

En machine learning classique, on extrait souvent les features manuellement.

Exemple :

```text
audio
↓
MFCC, ZCR, RMS, spectral centroid
↓
Random Forest / SVM / Logistic Regression
```

En deep learning, le modèle peut apprendre lui-même les features.

Exemple :

```text
audio
↓
spectrogramme
↓
CNN
↓
features apprises automatiquement
↓
classe prédite
```

Ou :

```text
audio brut
↓
réseau de neurones
↓
features internes apprises
↓
classe prédite
```

Différence :

| Approche                   | Features                                  |
| -------------------------- | ----------------------------------------- |
| Machine learning classique | Features extraites manuellement           |
| Deep learning              | Features souvent apprises automatiquement |
| Approche hybride           | Spectrogramme ou MFCC donnés au réseau    |

# 12. Qu’est-ce qu’un label ?

Un **label** est la réponse correcte associée à un exemple du dataset.

Exemple :

```text
chien_001.wav → chien
chien_002.wav → chien
chat_001.wav  → chat
pluie_001.wav → pluie
```

Le label sert pendant l’entraînement.

Le modèle fait une prédiction, puis on compare cette prédiction au label correct.

```text
features audio
↓
modèle
↓
prédiction : chat
↓
label réel : chien
↓
erreur
↓
correction du modèle
```

Le label est donc indispensable en apprentissage supervisé.

# 13. Le label dans différents types de problèmes

## 13.1 Classification

Dans une classification, le label est une catégorie.

Exemple :

```text
audio_001.wav → chien
audio_002.wav → chat
audio_003.wav → voiture
```

Ici, les labels sont discrets.

```text
chien
chat
voiture
```

## 13.2 Régression

Dans une régression, le label est une valeur numérique.

Exemple :

```text
audio_001.wav → 72 BPM
audio_002.wav → 130 BPM
audio_003.wav → 0.82 niveau d’énergie
```

Ici, le modèle ne prédit pas une classe, mais une valeur.

Exemples en audio :

```text
prédire le tempo
prédire l’âge d’un locuteur
prédire l’intensité émotionnelle
prédire la durée d’un son
prédire une note de qualité audio
```

## 13.3 Multi-label

Dans certains cas, un fichier audio peut avoir plusieurs labels.

Exemple :

```text
audio_001.wav → voix, musique, bruit de fond
audio_002.wav → voiture, pluie, klaxon
audio_003.wav → guitare, batterie, voix
```

Ce n’est pas une seule classe, mais plusieurs labels en même temps.

Très fréquent en audio réel.

Exemple :

```text
enregistrement de rue
↓
labels :
- voiture
- voix humaine
- sirène
- vent
```

## 13.4 Séquence de labels

Dans la reconnaissance vocale, le label peut être une séquence de mots.

Exemple :

```text
audio.wav → "bonjour je m'appelle Karim"
```

Ici, le label n’est pas une simple classe, mais une transcription.

En musique, le label peut aussi être une séquence :

```text
audio.wav → Do, Mi, Sol, Do
```

# 14. Qu’est-ce qu’une classe ?

Une **classe** est une catégorie possible que le modèle peut prédire.

Exemple :

```text
Problème : reconnaître un animal à partir d’un son

Classes :
- chien
- chat
- oiseau
- cheval
```

La classe est donc une option possible dans le problème.

## 14.1 Classe versus label

La différence est subtile mais importante.

```text
classe = catégorie possible
label = catégorie correcte d’un exemple
```

Exemple :

```text
Classes possibles :
chien, chat, oiseau

Fichier :
audio_001.wav

Label :
chien
```

Ici, `chien` est :

```text
une classe du problème
le label correct de audio_001.wav
```

Donc un même mot peut être à la fois une classe et un label selon le contexte.

# 15. Exemple très complet

Imaginons un projet : classifier des sons urbains.

## 15.1 Classes du projet

```text
voiture
klaxon
sirène
chien
pluie
marteau-piqueur
voix humaine
moto
```

Ces classes sont définies avant l’entraînement.

Elles représentent les catégories que le modèle devra reconnaître.

## 15.2 Dataset

```text
audio_001.wav → voiture
audio_002.wav → klaxon
audio_003.wav → chien
audio_004.wav → pluie
audio_005.wav → sirène
```

Chaque fichier possède un label.

## 15.3 Extraction de features

Pour chaque fichier, on extrait :

```text
MFCC
RMS Energy
Zero Crossing Rate
Spectral Centroid
Spectral Bandwidth
Mel Spectrogram
durée
fréquence dominante
```

Exemple :

```text
audio_001.wav
label : voiture

features :
RMS = 0.32
ZCR = 0.09
Spectral Centroid = 1450 Hz
Spectral Bandwidth = 2100 Hz
MFCC_1 = -180.4
MFCC_2 = 95.7
MFCC_3 = -24.1
...
```

Le modèle apprend à associer ces valeurs à la classe `voiture`.

# 16. Structure d’un dataset audio

Un dataset audio peut être organisé comme ceci :

```text
dataset/
│
├── chien/
│   ├── chien_001.wav
│   ├── chien_002.wav
│   └── chien_003.wav
│
├── chat/
│   ├── chat_001.wav
│   ├── chat_002.wav
│   └── chat_003.wav
│
├── pluie/
│   ├── pluie_001.wav
│   ├── pluie_002.wav
│   └── pluie_003.wav
│
└── voiture/
    ├── voiture_001.wav
    ├── voiture_002.wav
    └── voiture_003.wav
```

Ici, le nom du dossier peut servir de label.

Exemple :

```text
dataset/chien/chien_001.wav → label chien
dataset/chat/chat_001.wav   → label chat
dataset/pluie/pluie_001.wav → label pluie
```

Autre organisation possible :

```text
dataset/
├── audio/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── audio_003.wav
│
└── labels.csv
```

Avec un fichier CSV :

```text
filename,label
audio_001.wav,chien
audio_002.wav,chat
audio_003.wav,pluie
```

# 17. Encodage des labels

Les modèles travaillent avec des nombres.

Donc les labels texte doivent souvent être transformés en valeurs numériques.

Exemple :

```text
chien   → 0
chat    → 1
pluie   → 2
voiture → 3
```

Le modèle ne manipule pas directement le mot `chien`.

Il manipule plutôt :

```text
0
1
2
3
```

## 17.1 Label Encoding

Le label encoding transforme chaque classe en entier.

```text
chien   = 0
chat    = 1
pluie   = 2
voiture = 3
```

Attention : ce n’est pas parce que `voiture = 3` que voiture est “plus grand” que chien.

Ce sont juste des identifiants numériques.

## 17.2 One-hot encoding

Le one-hot encoding représente chaque classe par un vecteur.

Exemple avec 4 classes :

```text
chien   → [1, 0, 0, 0]
chat    → [0, 1, 0, 0]
pluie   → [0, 0, 1, 0]
voiture → [0, 0, 0, 1]
```

C’est très utilisé avec les réseaux de neurones.

# 18. Prédiction de classe

Pendant l’entraînement, le modèle apprend à prédire une classe.

Exemple :

```text
Entrée :
features de audio_001.wav

Sortie du modèle :
chien   : 0.80
chat    : 0.10
pluie   : 0.05
voiture : 0.05
```

Le modèle prédit la classe avec la probabilité la plus élevée.

```text
classe prédite = chien
```

Si le vrai label est `chien`, la prédiction est correcte.

Si le vrai label est `chat`, la prédiction est fausse.

# 19. Classe prédite et label réel

Il faut bien distinguer :

```text
label réel = vérité du dataset
classe prédite = réponse du modèle
```

Exemple :

```text
Fichier : audio_042.wav

Label réel :
chien

Prédiction du modèle :
chat
```

Ici, le modèle s’est trompé.

Autre exemple :

```text
Fichier : audio_043.wav

Label réel :
pluie

Prédiction du modèle :
pluie
```

Ici, le modèle a raison.

# 20. Erreur, perte et apprentissage

Pendant l’entraînement, le modèle compare :

```text
prédiction
label réel
```

S’il se trompe, on calcule une erreur.

Cette erreur sert à ajuster le modèle.

```text
features
↓
modèle
↓
prédiction
↓
comparaison avec le label
↓
erreur
↓
correction des poids du modèle
```

Exemple :

```text
Label réel :
chien

Prédiction :
chien : 0.40
chat  : 0.45
pluie : 0.10
voiture : 0.05
```

Le modèle pense que c’est plutôt un chat.

Comme le label réel est `chien`, on corrige le modèle pour augmenter la probabilité de `chien` et réduire celle de `chat`.

# 21. Exemple complet avec tableau

| Fichier       | Features extraites | Label réel | Classes possibles  | Classe prédite |
| ------------- | ------------------ | ---------- | ------------------ | -------------- |
| audio_001.wav | MFCC, RMS, ZCR     | chien      | chien, chat, pluie | chien          |
| audio_002.wav | MFCC, RMS, ZCR     | chat       | chien, chat, pluie | chien          |
| audio_003.wav | MFCC, RMS, ZCR     | pluie      | chien, chat, pluie | pluie          |

Analyse :

```text
audio_001.wav → correct
audio_002.wav → erreur
audio_003.wav → correct
```

# 22. Feature vector

Un **feature vector** est un vecteur contenant plusieurs features.

Exemple :

```text
audio_001.wav
↓
features :
[
  RMS,
  ZCR,
  Spectral Centroid,
  Spectral Bandwidth,
  MFCC_1,
  MFCC_2,
  MFCC_3,
  ...
]
```

Exemple numérique :

```text
[
  0.32,
  0.09,
  1450.0,
  2100.0,
  -180.4,
  95.7,
  -24.1
]
```

Ce vecteur devient l’entrée du modèle.

```text
feature vector
↓
modèle
↓
classe prédite
```

# 23. Feature matrix

Quand on a plusieurs fichiers, on obtient une matrice de features.

Exemple :

```text
audio_001.wav → [0.32, 0.09, 1450, -180, 95]
audio_002.wav → [0.21, 0.13, 2200, -150, 80]
audio_003.wav → [0.44, 0.05, 900, -230, 110]
```

On peut représenter cela comme une table :

| Fichier       |  RMS |  ZCR | Spectral Centroid | MFCC_1 | MFCC_2 | Label |
| ------------- | ---: | ---: | ----------------: | -----: | -----: | ----- |
| audio_001.wav | 0.32 | 0.09 |              1450 |   -180 |     95 | chien |
| audio_002.wav | 0.21 | 0.13 |              2200 |   -150 |     80 | chat  |
| audio_003.wav | 0.44 | 0.05 |               900 |   -230 |    110 | pluie |

La partie features est généralement appelée `X`.

La partie labels est généralement appelée `y`.

```text
X = features
y = labels
```

# 24. X et y en machine learning

En machine learning, on note souvent :

```text
X = données d’entrée
y = réponses attendues
```

Dans notre cas :

```text
X = features audio
y = labels
```

Exemple :

```text
X = [
  [0.32, 0.09, 1450],
  [0.21, 0.13, 2200],
  [0.44, 0.05, 900]
]

y = [
  "chien",
  "chat",
  "pluie"
]
```

Chaque ligne de `X` correspond à un fichier audio.

Chaque élément de `y` correspond au label de la même ligne.

```text
X[0] correspond à y[0]
X[1] correspond à y[1]
X[2] correspond à y[2]
```

# 25. Train, validation, test

On ne doit pas entraîner et tester le modèle sur exactement les mêmes fichiers.

On sépare généralement le dataset :

```text
train set       → pour apprendre
validation set  → pour régler le modèle
test set        → pour évaluer à la fin
```

Exemple :

```text
70 % entraînement
15 % validation
15 % test
```

## 25.1 Train set

Le modèle apprend avec ces données.

```text
features train + labels train
↓
apprentissage
```

## 25.2 Validation set

On utilise la validation pour vérifier si le modèle progresse correctement.

```text
modèle entraîné
↓
validation
↓
réglage des paramètres
```

## 25.3 Test set

Le test set sert à mesurer la performance finale.

Le modèle ne doit pas avoir vu ces fichiers pendant l’entraînement.

```text
modèle final
↓
test set
↓
score final
```

# 26. Les classes déséquilibrées

Un problème fréquent : certaines classes ont beaucoup plus d’exemples que d’autres.

Exemple :

```text
chien   → 10 000 fichiers
chat    → 9 000 fichiers
sirène  → 300 fichiers
klaxon  → 150 fichiers
```

Le modèle risque d’apprendre surtout les classes majoritaires.

Il peut devenir mauvais sur les classes rares.

## 26.1 Exemple de problème

Si le dataset contient :

```text
95 % de sons "voiture"
5 % de sons "sirène"
```

Un modèle naïf peut prédire `voiture` presque tout le temps et obtenir un bon score global, mais être inutile pour détecter les sirènes.

## 26.2 Solutions

On peut :

```text
ajouter des données pour les classes rares
rééquilibrer le dataset
utiliser de l’augmentation audio
pondérer la fonction de perte
utiliser des métriques adaptées
```

# 27. Data augmentation audio

La data augmentation consiste à modifier légèrement les fichiers audio pour créer de nouveaux exemples.

Exemples :

```text
ajouter du bruit
changer légèrement le pitch
modifier la vitesse
décaler le signal dans le temps
changer le volume
couper des segments
simuler une réverbération
```

Objectif :

```text
rendre le modèle plus robuste
augmenter la quantité de données
éviter le surapprentissage
```

Exemple :

```text
chien_001.wav
↓
version avec bruit
version plus rapide
version plus lente
version volume faible
version volume fort
```

Tous ces fichiers gardent normalement le même label `chien`.

# 28. Cas important : mauvaise qualité des labels

Les labels doivent être fiables.

Si les labels sont faux, le modèle apprend mal.

Exemple :

```text
audio_001.wav contient un chien
label donné : chat
```

Le modèle va apprendre une mauvaise association.

Problèmes possibles :

```text
erreur humaine d’annotation
fichier mal nommé
son contenant plusieurs sources
label trop vague
classe ambiguë
mauvaise segmentation audio
```

# 29. Label ambigu

Certains fichiers audio peuvent être difficiles à classer.

Exemple :

```text
un fichier contient :
- voiture
- pluie
- voix humaine
```

Si le dataset force un seul label, on perd de l’information.

Dans ce cas, il vaut mieux faire du multi-label :

```text
audio_001.wav → voiture, pluie, voix humaine
```

# 30. Classe trop générale ou trop précise

Le choix des classes est très important.

## 30.1 Classes trop générales

Exemple :

```text
animal
véhicule
nature
humain
```

C’est simple, mais peu précis.

Le modèle peut dire :

```text
animal
```

Mais il ne dira pas si c’est :

```text
chien
chat
oiseau
cheval
```

## 30.2 Classes trop précises

Exemple :

```text
chien berger allemand qui aboie fort
chien labrador qui aboie doucement
chien petit qui grogne
```

C’est très précis, mais plus difficile à apprendre.

Il faut beaucoup de données pour chaque classe.

## 30.3 Bon équilibre

Un bon projet choisit des classes :

```text
claires
utiles
séparables
suffisamment représentées
compréhensibles
```

# 31. Exemple pédagogique complet

Projet : reconnaître des sons dans une maison.

## 31.1 Classes

```text
porte
alarme
voix
chien
silence
aspirateur
musique
eau
```

## 31.2 Labels

```text
audio_001.wav → porte
audio_002.wav → aspirateur
audio_003.wav → voix
audio_004.wav → chien
audio_005.wav → eau
```

## 31.3 Features possibles

```text
RMS Energy
Zero Crossing Rate
MFCC
Mel Spectrogram
Spectral Centroid
Spectral Bandwidth
Durée
Pitch
```

## 31.4 Exemple d’apprentissage

```text
features de audio_001.wav → label porte
features de audio_002.wav → label aspirateur
features de audio_003.wav → label voix
```

## 31.5 Prédiction

```text
nouveau_audio.wav
↓
extraction des features
↓
modèle
↓
probabilités :
porte      : 0.04
alarme     : 0.02
voix       : 0.10
chien      : 0.78
silence    : 0.01
aspirateur : 0.03
musique    : 0.01
eau        : 0.01
↓
classe prédite : chien
```

# 32. Les métriques de performance

Une fois le modèle entraîné, on évalue ses résultats.

## 32.1 Accuracy

L’accuracy mesure le pourcentage de bonnes prédictions.

```text
accuracy = bonnes prédictions / total des prédictions
```

Exemple :

```text
100 fichiers testés
85 bien prédits
accuracy = 85 %
```

Attention : l’accuracy peut être trompeuse si les classes sont déséquilibrées.

## 32.2 Precision

La precision répond à la question :

```text
Quand le modèle prédit une classe, est-ce qu’il a souvent raison ?
```

Exemple :

```text
Le modèle prédit "sirène" 10 fois.
Il a raison 8 fois.
Precision = 80 %
```

## 32.3 Recall

Le recall répond à la question :

```text
Parmi tous les vrais exemples d’une classe, combien le modèle en retrouve ?
```

Exemple :

```text
Il y a 20 vraies sirènes.
Le modèle en détecte 14.
Recall = 70 %
```

## 32.4 F1-score

Le F1-score combine precision et recall.

Il est utile quand les classes sont déséquilibrées.

```text
bon F1-score = bon équilibre entre precision et recall
```

## 32.5 Matrice de confusion

La matrice de confusion montre où le modèle se trompe.

Exemple :

| Réel / Prédit | chien | chat | pluie |
| ------------- | ----: | ---: | ----: |
| chien         |    30 |    5 |     0 |
| chat          |     4 |   26 |     1 |
| pluie         |     0 |    2 |    32 |

Lecture :

```text
30 chiens bien reconnus comme chien
5 chiens confondus avec chat
4 chats confondus avec chien
32 pluies bien reconnues comme pluie
```

Très utile pour comprendre les erreurs.

# 33. Exemple de confusion audio

Certaines classes peuvent se ressembler.

Exemples :

```text
chien et loup
pluie et bruit blanc
voiture et moto
voix et chant
sirène et alarme
piano et guitare douce
```

Le modèle peut confondre ces classes si leurs features sont proches.

# 34. Résumé détaillé des notions

## Feature

Une feature est une caractéristique extraite du son.

Exemples :

```text
RMS Energy
Zero Crossing Rate
MFCC
Mel Spectrogram
Spectral Centroid
Pitch
Tempo
Chroma
```

Elle sert d’entrée au modèle.

## Label

Un label est la bonne réponse associée à un fichier.

Exemple :

```text
audio_001.wav → chien
```

Il sert à corriger le modèle pendant l’entraînement.

## Classe

Une classe est une catégorie possible.

Exemple :

```text
chien
chat
pluie
voiture
```

Le label d’un fichier est une des classes possibles.

## Prédiction

La prédiction est la réponse donnée par le modèle.

Exemple :

```text
classe prédite : chien
```

## Erreur

L’erreur est l’écart entre :

```text
label réel
classe prédite
```

Exemple :

```text
label réel : chat
classe prédite : chien
```

# 35. Schéma final complet

```text
1. On définit le problème
   Exemple : reconnaître des sons urbains

2. On définit les classes
   chien, voiture, sirène, pluie, voix

3. On collecte les fichiers audio
   audio_001.wav, audio_002.wav, audio_003.wav

4. On attribue les labels
   audio_001.wav → chien
   audio_002.wav → voiture
   audio_003.wav → pluie

5. On extrait les features
   MFCC, RMS, ZCR, spectrogramme, pitch

6. On construit X et y
   X = features
   y = labels

7. On entraîne le modèle
   X_train, y_train

8. Le modèle apprend les associations
   features de chien → classe chien
   features de pluie → classe pluie

9. On teste le modèle
   X_test

10. Le modèle prédit une classe
   nouveau_audio.wav → chien

11. On compare avec le label réel
   si label réel = chien → correct
   sinon → erreur
```

# 36. À retenir

Dans un projet de machine learning audio :

```text
Le fichier audio est la matière première.
Les features sont les informations extraites du son.
Les labels sont les bonnes réponses.
Les classes sont les catégories possibles.
Le modèle apprend à relier les features aux labels.
Une fois entraîné, il prédit une classe pour un nouveau son.
```

Exemple final :

```text
audio_chien.wav
↓
features :
MFCC, énergie, pitch, spectrogramme
↓
label réel :
chien
↓
classe possible :
chien
↓
modèle entraîné
↓
prédiction :
chien
```

La phrase la plus importante à retenir :

```text
En audio processing, on transforme le son en features numériques, puis le modèle apprend à associer ces features à une classe grâce aux labels.
```
