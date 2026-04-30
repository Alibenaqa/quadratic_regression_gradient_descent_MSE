# Compte rendu — TP Régression quadratique (MSE)
Ali Benaqa

---

## PARTIE A

**1) Pourquoi la standardisation aide la descente de gradient ?**
La standardisation met toutes les variables sur la même échelle, ce qui stabilise et accélère la convergence de la descente de gradient.

**2) Que se passe-t-il si on ne normalise pas (surfaces en dizaines, prix en centaines de milliers) ?**
Le gradient lié au prix serait beaucoup plus grand que celui de la surface. Le learning rate serait inadapté pour l'une des deux variables : la descente de gradient divergerait ou convergerait très lentement.

**3) La relation semble-t-elle parfaitement linéaire ? Que pourrait apporter un modèle quadratique ?**
La relation est quasiment linéaire mais pas tout à fait. Un modèle quadratique peut introduire une légère courbure pour mieux s'ajuster aux données.

---

## PARTIE B

**1) Quelles sont les trois poids du modèle ?**
Les 3 poids sont : a, b, c dans ŷ = ax² + bx + c.

**2) En quoi ce modèle est-il plus flexible qu'un modèle affine ?**
Contrairement à une droite, le modèle quadratique peut former une courbure. Il s'adapte mieux à des jeux de données dont la relation n'est pas strictement linéaire.

**3) Quelle différence entre MSE et RMSE ? Pourquoi la RMSE est plus lisible ?**
La MSE est exprimée en unités au carré (erreur quadratique), la RMSE est sa racine carrée donc dans la même unité que la variable cible. Elle est plus lisible car directement interprétable dans le contexte d'étude.

---

## PARTIE C

**1) Quelle est la dérivée de (ŷ − y)² par rapport à ŷ ?**
2(ŷ − y) = 2eᵢ

**2) Quelle est la dérivée de ŷ = ax² + bx + c par rapport à a, b, c ?**
- Par rapport à a : xᵢ²
- Par rapport à b : xᵢ
- Par rapport à c : 1

**3) Où intervient le facteur 1/n ?**
Le facteur 1/n vient de la définition de la MSE (Mean Squared Error) qui est une moyenne des erreurs au carré sur les n exemples.

Les gradients obtenus par la règle de dérivation en chaîne sont donc :

    ∂L/∂a = (2/n) Σ eᵢ xᵢ²
    ∂L/∂b = (2/n) Σ eᵢ xᵢ
    ∂L/∂c = (2/n) Σ eᵢ

---

## PARTIE D

**1) À quoi sert le learning rate η ?**
Il contrôle la taille du pas effectué à chaque itération dans le sens opposé au gradient. Plus il est petit, plus on converge précisément mais lentement ; plus il est grand, plus on risque de dépasser le minimum.

**2) Deux symptômes d'un learning rate trop grand :**
- La RMSE oscille ou diverge, sans jamais se stabiliser.
- Les paramètres dépassent le minimum et "rebondissent" sans converger.

**3) Symptôme d'un learning rate trop petit :**
- La convergence est extrêmement lente, le temps de calcul devient prohibitif.

---

## PARTIE E

**1) La RMSE doit-elle être strictement décroissante à chaque epoch ?**
Non, elle doit être globalement décroissante. Si elle oscille fortement, cela indique un problème avec le learning rate. Plus elle diminue, plus le modèle converge vers une prédiction propre.

**2) Quand peut-on arrêter l'entraînement (early stopping) ?**
On peut arrêter lorsque l'amélioration de la RMSE entre deux epochs consécutives devient inférieure à un seuil (ex. 1e-6), ce qui signifie qu'il n'y aura plus d'amélioration significative.

---

## PARTIE F

**1) Le modèle quadratique fait-il toujours mieux ?**
Non. Dans notre cas, la courbure reste minime car les données forment naturellement une droite. Le gain en RMSE est faible (~1 % : 0.1076 vs 0.1065). De plus, le modèle quadratique, ayant un paramètre supplémentaire, risque de faire de l'overfitting : il peut enregistrer les variations aléatoires du jeu d'entraînement et généraliser moins bien sur de nouvelles données. La régression linéaire reste plus robuste dans ce contexte.
