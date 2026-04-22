# Jour 2

## 1. Appliquer le machine learning a la classification de genres

**Introduction**
Cette partie montre comment utiliser les features audio pour entrainer un modele de classification.

**Explication**
L'idee est de representer chaque morceau par un vecteur de caracteristiques, puis d'apprendre a associer ce vecteur a un genre.

**Contexte**
On peut utiliser cette approche pour categoriser automatiquement une bibliotheque musicale.

**Formule mathematique**

$$
\hat{y} = \arg\max_{k} \, p(y=k \mid \mathbf{x})
$$

**Lecture de la formule**
"y chapeau egale arg max sur k de p de y egal k sachant x."

**Sens de la formule**
Le modele choisit la classe la plus probable a partir des caracteristiques du morceau.

**Decomposition mathematique**
- `\mathbf{x}` : vecteur de features audio
- `y` : classe ou genre
- `\hat{y}` : classe predite

**Resultat attendu**
L'etudiant sait entrainer et evaluer un modele supervise simple sur des donnees audio.

**Code**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## 2. Construire un moteur de recommandation musicale

**Introduction**
Cette partie utilise la similarite entre morceaux pour recommander des titres proches.

**Explication**
On compare les vecteurs de features de plusieurs morceaux et on cherche ceux qui sont les plus proches du morceau cible.

**Contexte**
C'est le coeur du projet final de type Spotify-like centree sur la decouverte musicale.

**Formule mathematique**

$$
d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{i=1}^{d} (x_i - z_i)^2}
$$

**Lecture de la formule**
"d de x et z egale racine carree de la somme pour i allant de 1 a d de x i moins z i au carre."

**Sens de la formule**
Cette distance mesure a quel point deux morceaux sont proches dans l'espace des features.

**Decomposition mathematique**
- `\mathbf{x}` : vecteur du morceau de reference
- `\mathbf{z}` : vecteur du morceau compare
- `d` : dimension de l'espace de caracteristiques

**Resultat attendu**
L'etudiant sait construire un systeme simple de recommandation a partir de descripteurs audio.

**Code**

```python
from sklearn.metrics.pairwise import euclidean_distances

distances = euclidean_distances([target_vector], feature_matrix)[0]
top_indices = distances.argsort()[:5]

print(top_indices)
```

## Synthese du jour

- Preparer des donnees audio pour le ML.
- Entrainer et evaluer un classifieur.
- Recommander des morceaux par distance dans l'espace des features.
