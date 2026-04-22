# Jour 2

**Notions clés**
- **Jeu de données** : collection de morceaux décrits par des features audio.
- **Features** : variables numériques qui résument le son.
- **Labels** : genres ou classes associées aux morceaux.
- **Train/Test split** : séparation des données pour apprendre puis évaluer.
- **Classifieur** : modèle qui prédit une classe à partir de features.
- **Matrice de confusion** : tableau qui compare les classes réelles et prédites.
- **Similarité** : proximité entre deux morceaux dans l'espace des features.

## Appliquer le machine learning à la classification de genres (3h30)

**Introduction**
Cette partie montre comment utiliser les features audio pour entraîner un modèle de classification.
L'objectif est de passer d'un tableau de descripteurs à une prédiction de genre exploitable.

**Explication**
L'idée est de représenter chaque morceau par un vecteur de caractéristiques, puis d'apprendre à associer ce vecteur à un genre.
En pratique, chaque ligne du dataset devient un morceau, et chaque colonne représente une feature audio différente.

**Pourquoi cette étape est indispensable ?**
Sans préparation du dataset, un modèle ne sait pas lire la structure musicale. Les features jouent le rôle de langage d'entrée pour le classifieur.

**Contexte**
On peut utiliser cette approche pour catégoriser automatiquement une bibliothèque musicale.
Elle sert aussi à vérifier si certains genres sont faciles à distinguer à partir de leurs caractéristiques sonores.

**Étapes à retenir**
1. Construire la matrice de features `X`.
2. Construire le vecteur de labels `y`.
3. Séparer les données en apprentissage et en test.
4. Entraîner le modèle.
5. Mesurer ses erreurs.

**Formule mathematique**

$$
\hat{y} = \arg\max_{k} \, p(y=k \mid \mathbf{x})
$$

**Lecture de la formule**
"y chapeau égale arg max sur k de p de y égal k sachant x."

**Sens de la formule**
Le modèle choisit la classe la plus probable à partir des caractéristiques du morceau.
Autrement dit, le classifieur transforme les features en décision de genre.

**Lien avec le code**
Dans le code, `fit` correspond à l'apprentissage de cette règle, puis `predict` applique cette logique à de nouveaux morceaux.

**Décomposition mathématique**
- `\mathbf{x}` : vecteur de features audio
- `y` : classe ou genre
- `\hat{y}` : classe prédite

**Pourquoi Random Forest ?**
Random Forest est robuste, simple à utiliser et permet déjà d'obtenir un bon point de départ sur des features audio tabulaires.

**Pourquoi tester d'autres modèles ?**
k-NN, Random Forest ou d'autres classifieurs ne réagissent pas pareil aux mêmes features. Le but est de comparer leurs performances sur la même base de données.

**Comparer les performances des algorithmes**
On ne choisit pas un modèle uniquement parce qu'il fonctionne. On le compare avec d'autres à l'aide de métriques comme la précision, le rappel, le score F1 et la matrice de confusion.

**Résultat attendu**
Savoir entraîner et évaluer un modèle supervisé simple sur des données audio.
Savoir expliquer comment le dataset est préparé et pourquoi la séparation train/test est nécessaire.
Savoir lire une matrice de confusion et comprendre les erreurs du modèle.

**Code**

```python
# Decoupage des donnees et entrainement d'un classifieur.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Separation train / test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modele de classification.
model = RandomForestClassifier(random_state=42)
# Apprentissage sur les donnees d'entrainement.
model.fit(X_train, y_train)

# Prediction sur les donnees de test.
y_pred = model.predict(X_test)
# Evaluation des performances.
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Explication du code**
Ce bloc montre le cycle classique du machine learning supervisé : séparation des données, apprentissage du modèle, puis évaluation.
La matrice de confusion permet de voir où le modèle confond certains genres, tandis que le rapport de classification résume précision, rappel et score F1.

**Ce qu'il faut observer**
- si le modèle généralise bien sur les données de test ;
- quels genres sont souvent confondus ;
- si certaines classes sont déséquilibrées ou plus difficiles à prédire.

## Construire un moteur de recommandation musicale (3h30)

**Introduction**
Cette partie utilise la similarité entre morceaux pour recommander des titres proches.
Le principe est de retrouver, dans l'espace des features, les morceaux les plus voisins d'un morceau de référence.

**Explication**
On compare les vecteurs de features de plusieurs morceaux et on cherche ceux qui sont les plus proches du morceau cible.
La recommandation ne repose donc pas sur un genre imposé, mais sur une proximité acoustique mesurable.

**Contexte**
C'est le cœur du projet final de type Spotify-like centré sur la découverte musicale.
Dans une version plus avancée, cette logique peut être combinée à l'API Spotify pour récupérer les métadonnées, les pochettes ou les aperçus audio.

**Étapes à retenir**
1. Représenter chaque morceau par un vecteur de features.
2. Choisir un morceau de référence.
3. Calculer la distance entre ce morceau et tous les autres.
4. Trier les distances.
5. Garder les 5 morceaux les plus proches.

**Formule mathematique**

$$
d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{i=1}^{d} (x_i - z_i)^2}
$$

**Lecture de la formule**
"d de x et z égale racine carrée de la somme pour i allant de 1 à d de x i moins z i au carré."

**Sens de la formule**
Cette distance mesure à quel point deux morceaux sont proches dans l'espace des features.
Plus la distance est petite, plus les morceaux se ressemblent selon les descripteurs choisis.

**Lien avec le code**
`euclidean_distances` calcule cette distance pour chaque morceau de la base, puis `argsort` permet de récupérer les indices des plus proches.

**Décomposition mathématique**
- `\mathbf{x}` : vecteur du morceau de référence
- `\mathbf{z}` : vecteur du morceau comparé
- `d` : dimension de l'espace de caractéristiques

**Résultat attendu**
Savoir construire un système simple de recommandation à partir de descripteurs audio.
Savoir expliquer pourquoi la distance permet de remplacer une comparaison manuelle entre morceaux.
Savoir justifier le choix des 5 recommandations les plus proches.

**Code**

```python
# Mesure de distance entre morceaux.
from sklearn.metrics.pairwise import euclidean_distances

# Calcul des distances entre un morceau cible et la base de morceaux.
distances = euclidean_distances([target_vector], feature_matrix)[0]
# Selection des morceaux les plus proches.
top_indices = distances.argsort()[:5]

# Affichage des indices recommandes.
print(top_indices)
```

**Explication du code**
Ce bloc implémente une recommandation par similarité.
Plus la distance est faible, plus deux morceaux sont proches dans l'espace des features, ce qui soutient la logique de découverte musicale.
Dans le projet final, ces indices pourront ensuite être reliés aux titres récupérés via l'API Spotify.

## Synthèse du jour

- Préparer des données audio pour le ML.
- Entraîner et évaluer un classifieur.
- Recommander des morceaux par distance dans l'espace des features.
