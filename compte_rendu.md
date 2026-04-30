# Compte rendu — TP Régression quadratique (MSE)

## 1. Dérivation des gradients

On définit l'erreur sur le i-ème exemple :

$$e_i = \hat{y}_i - y_i \quad \text{avec} \quad \hat{y}_i = ax_i^2 + bx_i + c$$

La loss MSE est :

$$L(a, b, c) = \frac{1}{n} \sum_{i=1}^{n} e_i^2$$

En appliquant la règle de dérivation en chaîne :

$$\frac{\partial L}{\partial a} = \frac{1}{n} \sum_{i=1}^{n} 2e_i \cdot \frac{\partial \hat{y}_i}{\partial a} = \frac{2}{n} \sum_{i=1}^{n} e_i x_i^2$$

$$\frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} e_i x_i$$

$$\frac{\partial L}{\partial c} = \frac{2}{n} \sum_{i=1}^{n} e_i$$

Le facteur $\frac{1}{n}$ provient de la moyenne dans la MSE. La dérivée de $e_i^2$ par rapport à $\hat{y}_i$ donne $2e_i$, et les dérivées partielles de $\hat{y}_i$ par rapport à $a$, $b$, $c$ donnent respectivement $x_i^2$, $x_i$, $1$.

## 2. Comparaison linéaire vs quadratique

| Modèle       | RMSE finale (données normalisées) |
|--------------|-----------------------------------|
| Linéaire     | 0.1076                            |
| Quadratique  | 0.1065                            |

Le modèle quadratique obtient une RMSE légèrement inférieure, ce qui indique un meilleur ajustement. Visuellement, la courbe quadratique épouse mieux la légère courbure du nuage de points.

Cependant, le gain est faible (~1 %), car la relation surface → prix est quasi-linéaire dans ce dataset. Le modèle quadratique n'apporte pas un avantage décisif ici, et présente un risque d'**overfitting** plus élevé : avec un paramètre supplémentaire, il peut s'adapter au bruit plutôt qu'à la tendance réelle, surtout sur de petits jeux de données.

## 3. Commentaire sur le learning rate

Le learning rate $\eta = 0.1$ a été utilisé avec 1 000 epochs. La RMSE converge de façon régulière et décroissante dès les premières epochs, ce qui confirme que le learning rate est bien calibré.

- **Learning rate trop grand** : la RMSE oscille ou diverge (les mises à jour dépassent le minimum).
- **Learning rate trop petit** : la convergence est très lente, le modèle n'atteint pas le minimum en un nombre raisonnable d'epochs.

La standardisation des données (moyenne 0, écart-type 1) est essentielle pour que ce learning rate unique fonctionne bien sur $x$ et $y$ d'échelles très différentes (surfaces en m², prix en centaines de milliers d'euros).
