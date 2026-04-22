# Gabarit — Section de cours ML-DL

## Modele de section

```md
### X.Y Titre de la notion

**Introduction**
Ecrire en 1 a 3 phrases ce qu'est la notion et son objectif principal.

**Explication**
Donner l'intuition du concept avec une image mentale, une analogie ou une logique simple.

**Contexte**
Donner un cas reel ou metier ou cette notion est utile.

**Formule mathematique**

$$
Ecrire ici la formule centrale
$$

**Lecture de la formule**
"Lire ici la formule comme un mathématicien la prononcerait à l'oral."

**Sens de la formule**
Expliquer ce que la formule exprime globalement.

**Décomposition mathématique**
- terme 1 : définition
- terme 2 : définition
- terme 3 : définition

**Résultat attendu**
Dire ce que le lecteur doit comprendre, visualiser ou savoir faire.

**Code**

```python
# code complet et executable
```

**Explication du code**
Expliquer en quelques phrases ce que fait le code, pourquoi il est utile et quel point theorique il illustre.
```

---

## Exemple court

```md
### 3.1 Intuition d'un mélange de gaussiennes

**Introduction**
Un GMM part du constat que des données réelles proviennent souvent de plusieurs sous-populations superposées, chacune gaussienne.

**Explication**
Imaginez une urne contenant des billes de plusieurs couleurs mélangées. On observe les billes, mais pas directement la couleur de la composante qui les a générées.

**Contexte**
En marketing, des revenus clients peuvent former plusieurs groupes : faibles, moyens et élevés.

**Formule mathematique**

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

**Lecture de la formule**
"p de x egale somme pour k allant de 1 a K de pi indice k N de x sachant mu indice k, Sigma indice k."

**Sens de la formule**
La densité observée est la somme pondérée de plusieurs gaussiennes, chacune représentant une sous-population.

**Décomposition mathématique**
- K : nombre de composantes
- pi indice k : poids de la composante k
- mu indice k : centre de la composante k
- Sigma indice k : covariance de la composante k

**Résultat attendu**
Comprendre qu'un même nuage de points peut être expliqué par plusieurs populations gaussiennes superposées.

**Code**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
```

**Explication du code**
Ce code charge les bibliotheques utiles pour generer, visualiser ou calculer des elements du melange de gaussiennes. L'objectif est de relier la formule a une implementation concrète.
```

---

## Regle pratique

Si un agent hesite sur la redaction, il doit toujours suivre cet ordre de priorite :

1. clarte pedagogique ;
2. rigueur mathematique ;
3. homogeneite du format ;
4. utilite pratique ;
5. completude du code.

## Regle d'ecriture

Conserver l'orthographe francaise correcte et tous les accents dans le rendu final.
