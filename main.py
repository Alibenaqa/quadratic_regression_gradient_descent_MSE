import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import load_data, standardize
from src.model import linear_predict, quadratic_predict
from src.train import gradient_descent_linear, gradient_descent_quadratic

DATA_PATH = "data/prix_maisons.csv"
LR = 0.1
EPOCHS = 1000


def plot_scatter_raw(surface, prix):
    plt.figure()
    plt.scatter(surface, prix, s=15)
    plt.xlabel("Surface (m²)")
    plt.ylabel("Prix (€)")
    plt.title("Surface vs Prix (données brutes)")
    plt.tight_layout()
    plt.savefig("plots/scatter_raw.png", dpi=150)
    plt.show()
    plt.close()


def plot_predictions(x, y, a_l, b_l, a_q, b_q, c_q):
    # Scatter + regression curves
    x_plot = np.linspace(x.min(), x.max(), 300)
    plt.figure()
    plt.scatter(x, y, s=15, label="Données")
    plt.plot(x_plot, linear_predict(a_l, b_l, x_plot), label="Linéaire")
    y_q = quadratic_predict(a_q, b_q, c_q, x_plot)
    plt.plot(x_plot, y_q, label="Quadratique")
    plt.xlabel("Surface normalisée")
    plt.ylabel("Prix normalisé")
    plt.title("Régression linéaire vs quadratique")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/regression_comparison.png", dpi=150)
    plt.show()
    plt.close()


def plot_rmse(hist_l, hist_q):
    # RMSE over epochs
    plt.figure()
    plt.plot(hist_l, label="Linéaire")
    plt.plot(hist_q, label="Quadratique")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Convergence RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/rmse_epochs.png", dpi=150)
    plt.show()
    plt.close()


def main():
    # Load and standardize
    surface, prix = load_data(DATA_PATH)
    x, _, _ = standardize(surface)
    y, _, _ = standardize(prix)

    # Train both models
    a_l, b_l, hist_l = gradient_descent_linear(x, y, LR, EPOCHS)
    a_q, b_q, c_q, hist_q = gradient_descent_quadratic(x, y, LR, EPOCHS)

    # Save figures
    plot_scatter_raw(surface, prix)
    plot_predictions(x, y, a_l, b_l, a_q, b_q, c_q)
    plot_rmse(hist_l, hist_q)

    print(f"RMSE linéaire final   : {hist_l[-1]:.4f}")
    print(f"RMSE quadratique final: {hist_q[-1]:.4f}")


if __name__ == "__main__":
    main()
