# Quadratic Regression — Gradient Descent (MSE)

TP — Entraînement supervisé : régression quadratique `ŷ = ax² + bx + c` par descente de gradient.

## Contexte

À partir d'un dataset de 100 maisons (surface en m², prix en €), on entraîne deux modèles de régression **sans librairie de ML** (NumPy uniquement) :

- **Linéaire** : `ŷ = ax + b`
- **Quadratique** : `ŷ = ax² + bx + c`

Les paramètres sont appris par **descente de gradient** : à chaque epoch, on calcule les gradients de la MSE par rapport à chaque paramètre, puis on les met à jour dans le sens opposé au gradient. Les données sont standardisées (moyenne 0, écart-type 1) pour stabiliser la convergence.

**Résultats obtenus** (1000 epochs, learning rate 0.1) :

| Modèle      | RMSE finale |
|-------------|-------------|
| Linéaire    | 0.1076      |
| Quadratique | 0.1065      |

Le modèle quadratique est légèrement meilleur, mais le gain est faible (~1 %) car la relation surface → prix est quasi-linéaire. Il présente un risque d'overfitting plus élevé.

## Structure

```
├── data/               # Dataset
├── plots/              # Figures générées
├── src/
│   ├── preprocessing.py  # Chargement et standardisation
│   ├── model.py          # Prédictions linéaire et quadratique
│   ├── loss.py           # MSE et RMSE
│   └── train.py          # Descente de gradient
└── main.py             # Pipeline complet
```

## Utilisation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Résultats

- `plots/scatter_raw.png` — nuage de points brut (surface vs prix)
- `plots/regression_comparison.png` — données normalisées + courbes linéaire/quadratique
- `plots/rmse_epochs.png` — convergence RMSE par epoch
