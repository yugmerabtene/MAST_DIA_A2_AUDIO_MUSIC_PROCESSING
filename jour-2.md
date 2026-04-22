# Jour 2

## 1. Appliquer le machine learning à la classification de genres

**Introduction**
Cette partie montre comment utiliser les features audio pour entraîner un modèle de classification.

**Explication**
L'idée est de représenter chaque morceau par un vecteur de caractéristiques, puis d'apprendre à associer ce vecteur à un genre.

**Contexte**
On peut utiliser cette approche pour catégoriser automatiquement une bibliothèque musicale.

**Formule mathematique**

$$
\hat{y} = \arg\max_{k} \, p(y=k \mid \mathbf{x})
$$

**Lecture de la formule**
"y chapeau égale arg max sur k de p de y égal k sachant x."

**Sens de la formule**
Le modèle choisit la classe la plus probable à partir des caractéristiques du morceau.

**Décomposition mathématique**
- `\mathbf{x}` : vecteur de features audio
- `y` : classe ou genre
- `\hat{y}` : classe prédite

**Résultat attendu**
Savoir entraîner et évaluer un modèle supervisé simple sur des données audio.

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
Ce bloc montre le cycle classique du machine learning supervise : separation des donnees, apprentissage du modele, puis evaluation. Il illustre comment les features audio deviennent une entree exploitable pour la classification.

## 2. Construire un moteur de recommandation musicale

**Introduction**
Cette partie utilise la similarité entre morceaux pour recommander des titres proches.

**Explication**
On compare les vecteurs de features de plusieurs morceaux et on cherche ceux qui sont les plus proches du morceau cible.

**Contexte**
C'est le cœur du projet final de type Spotify-like centré sur la découverte musicale.

**Formule mathematique**

$$
d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{i=1}^{d} (x_i - z_i)^2}
$$

**Lecture de la formule**
"d de x et z égale racine carrée de la somme pour i allant de 1 à d de x i moins z i au carré."

**Sens de la formule**
Cette distance mesure à quel point deux morceaux sont proches dans l'espace des features.

**Décomposition mathématique**
- `\mathbf{x}` : vecteur du morceau de référence
- `\mathbf{z}` : vecteur du morceau comparé
- `d` : dimension de l'espace de caractéristiques

**Résultat attendu**
Savoir construire un système simple de recommandation à partir de descripteurs audio.

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
Ce bloc implémente une recommandation par similarité. Plus la distance est faible, plus deux morceaux sont proches dans l'espace des features, ce qui soutient la logique de découverte musicale.

## Synthese du jour

- Préparer des données audio pour le ML.
- Entraîner et évaluer un classifieur.
- Recommander des morceaux par distance dans l'espace des features.
