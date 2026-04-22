# Lab 01 - Signaux audio et features

## Objectif

Manipuler un vrai fichier audio, visualiser sa forme d'onde et son spectrogramme, puis extraire quelques features de base.

Le fichier de référence du lab est `assets/Games.wav`.

## Fichier audio

- fichier : `assets/Games.wav`
- nature du fichier : extrait audio réel
- particularité : les scripts convertissent automatiquement le son en mono et n'analysent que les `20` premières secondes pour garder un affichage lisible et des calculs rapides

## Ressources pour récupérer un WAV

Pour tester avec d'autres sons, vous pouvez récupérer de l'audio depuis des plateformes publiques puis utiliser du `.wav`.

### Sources possibles

- **SoundCloud** : utile pour trouver des morceaux ou extraits, mais toujours vérifier les droits.
- **Freesound** : banque de sons avec licences indiquées.
- **Internet Archive (Audio)** : archives audio souvent réutilisables selon la licence.
- **Wikimedia Commons (Audio)** : médias avec conditions de licence explicites.
- **SampleSwap / Looperman** : échantillons audio pour pratique et tests.

### Vérification licence (obligatoire)

Avant usage en cours ou projet :

- vérifier la licence (CC0, CC-BY, etc.) ;
- vérifier l'usage pédagogique autorisé ;
- vérifier les obligations d'attribution ;
- éviter toute réutilisation non autorisée.

### Recommandation technique

Pour garder des labs stables :

- format `.wav` ;
- fréquence d'échantillonnage `44.1 kHz` ou `48 kHz` ;
- extrait court (10 à 30 secondes) ;
- mono ou stéréo (les scripts convertissent en mono).

## Scripts du lab

- `scripts/01_load_signal.py`
- `scripts/02_waveform_and_spectrogram.py`
- `scripts/03_audio_features.py`
- `scripts/04_project_ready_examples.py`

## Schéma du lab

```mermaid
flowchart LR
    A[Games.wav] --> B[Charger le signal]
    B --> C[Passer en mono]
    C --> D[Voir la forme d'onde]
    D --> E[Voir le spectrogramme]
    E --> F[Calculer les features]
    F --> G[Interpréter les résultats]
```

Ce schéma montre la logique du lab : on part d'un fichier audio brut, on le rend exploitable, puis on le transforme en représentations et en mesures numériques.

## Vocabulaire

- **Signal audio** : suite de valeurs qui représente le son dans le temps.
- **Mono** : un seul canal audio.
- **Stéréo** : deux canaux audio, généralement gauche et droite.
- **Forme d'onde** : lecture temporelle de l'amplitude.
- **Spectrogramme** : lecture temps-fréquence de l'énergie.
- **Feature** : mesure numérique compacte extraite du signal.
- **ZCR** : nombre de changements de signe du signal.
- **Centre spectral** : zone moyenne où l'énergie du signal est concentrée.
- **Bandwidth spectrale** : étendue de l'énergie autour du centre spectral.
- **Harmoniques** : fréquences liées à la note principale du son.

## Étape 1 - Charger le signal

**Résultat attendu**
Lire le fichier audio, connaître sa fréquence d'échantillonnage, son nombre de canaux et la durée analysée.

**Script**

```bash
python3 scripts/01_load_signal.py
```

**Ce qu'il faut observer**
- la fréquence d'échantillonnage ;
- la forme du tableau audio ;
- si le son est mono ou stéréo ;
- la durée réellement analysée.

**Ce que cela signifie**
Cette étape confirme que le fichier audio est correctement chargé et prêt pour les étapes suivantes.

## Étape 2 - Visualiser la forme d'onde et le spectrogramme

**Résultat attendu**
Créer deux représentations visuelles du son : la forme d'onde et le spectrogramme.

**Script**

```bash
python3 scripts/02_waveform_and_spectrogram.py
```

**Fichiers générés**
- `outputs/waveform.png`
- `outputs/spectrogram.png`

**Ce qu'il faut observer**
- dans la forme d'onde : les variations d'amplitude dans le temps ;
- dans le spectrogramme : les zones de fréquences les plus actives ;
- la différence entre une vue temporelle et une vue temps-fréquence.

**Ce que cela signifie**
La forme d'onde dit quand le son est fort ou faible. Le spectrogramme montre quelles fréquences dominent selon le temps.

## Étape 3 - Extraire des features audio

**Résultat attendu**
Calculer quelques nombres simples qui résument le comportement du son.

**Script**

```bash
python3 scripts/03_audio_features.py
```

**Ce qu'il faut observer**
- la valeur du `ZCR` ;
- la valeur du `centre spectral` ;
- la valeur de la `bandwidth spectrale` ;
- les principales fréquences dominantes affichées.

**Ce que cela signifie**
Ces nombres permettent de passer d'une écoute subjective du son à une description numérique exploitable pour l'analyse, la classification ou la recommandation.

## Étape 4 - Exemples orientés projet final

**Résultat attendu**
Construire un mini dataset de features, comparer des profils sonores, puis produire un premier ranking de similarité.

**Script**

```bash
python3 scripts/04_project_ready_examples.py
```

**Ce qu'il faut observer**
- la table de features par segment ;
- la distance entre deux segments ;
- l'ordre des segments les plus proches du segment de référence.

**Ce que cela signifie**
Cette étape simule en miniature ce que fera le projet final : créer des descripteurs, calculer des distances et recommander les éléments les plus similaires.

## Lien avec le cours

Ce lab illustre directement le chapitre `Comprendre la structure musicale et les signaux audio (3H30)` puis le chapitre `Extraire les caractéristiques audio (3h30)` du jour 1.

## Bilan du lab

- vérifier qu'un fichier audio réel est exploitable ;
- comprendre la différence entre signal, forme d'onde et spectrogramme ;
- extraire des features simples ;
- relier les mesures calculées à ce que l'on entend dans le fichier ;
- produire un premier ranking de similarité orienté recommandation.
