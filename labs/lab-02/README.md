# Lab 02 - Classification et recommandation

## Objectif

Passer d'un mini dataset de features audio a deux usages concrets du jour 2 :

- classification de genres (supervisee) ;
- recommandation musicale par similarite.

Le lab utilise `assets/mini_genre_dataset.csv`, un jeu de donnees pedagogique pret a executer.

## Fichier de donnees

- chemin : `assets/mini_genre_dataset.csv`
- colonnes : `track_id`, `genre`, `zcr`, `centroid_hz`, `bandwidth_hz`, `rms`
- taille : 18 morceaux, 3 genres

## Scripts du lab

- `scripts/01_genre_classification.py`
- `scripts/02_similarity_recommender.py`

## Etape 1 - Classification supervisee

**Resultat attendu**
Entrainer un classifieur de genres, puis lire les metriques principales.

**Script**

```bash
python3 scripts/01_genre_classification.py
```

**Ce qu'il faut observer**
- la taille du split train/test ;
- l'accuracy globale ;
- la matrice de confusion ;
- les scores precision/rappel/F1 par genre.

## Etape 2 - Recommandation par distance

**Resultat attendu**
Choisir un morceau cible et recuperer ses voisins les plus proches.

**Script**

```bash
python3 scripts/02_similarity_recommender.py
```

**Ce qu'il faut observer**
- le morceau cible (`TARGET_TRACK_ID`) ;
- les 5 recommandations les plus proches ;
- la distance associee a chaque recommandation.

## Lien avec le cours

Ce lab illustre directement les deux chapitres du jour 2 :

- `Appliquer le machine learning a la classification de genres (3h30)` ;
- `Construire un moteur de recommandation musicale (3h30)`.
