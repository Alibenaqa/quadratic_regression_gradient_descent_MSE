# Quadratic Regression — Gradient Descent (MSE)

TP — Entraînement supervisé : régression quadratique `ŷ = ax² + bx + c` par descente de gradient.

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
pip install -r requirements.txt
python main.py
```

## Résultats

- `plots/regression_comparison.png` — données + courbes linéaire/quadratique
- `plots/rmse_epochs.png` — convergence RMSE par epoch
