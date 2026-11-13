"""Hoeffding's D: dependencia no monotonica (comentario C34 del revisor).

D detecta cualquier tipo de dependencia entre dos variables, no solo la
monotonica. Rango aprox. [-0.5, 1]: 0 = independencia, 1 = dependencia total.
Implementacion directa de Hoeffding (1948), O(n^2).
"""
import numpy as np
from scipy.stats import rankdata


def hoeffding_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return np.nan
    R = rankdata(x); S = rankdata(y)
    # Q_i = 1 + #{j != i : x_j < x_i y y_j < y_i}, con medio punto por empates
    lt_x = (x[None, :] < x[:, None]).astype(float) + 0.5 * (x[None, :] == x[:, None])
    lt_y = (y[None, :] < y[:, None]).astype(float) + 0.5 * (y[None, :] == y[:, None])
    np.fill_diagonal(lt_x, 0.0); np.fill_diagonal(lt_y, 0.0)
    Q = 1.0 + (lt_x * lt_y).sum(axis=1)

    D1 = np.sum((Q - 1) * (Q - 2))
    D2 = np.sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
    D3 = np.sum((R - 2) * (S - 2) * (Q - 1))
    num = (n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3
    den = n * (n - 1) * (n - 2) * (n - 3) * (n - 4)
    return 30.0 * num / den


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-1, 1, n)
    casos = {
        "independiente": rng.uniform(-1, 1, n),
        "lineal":        2 * x + 0.05 * rng.normal(size=n),
        "monotona (exp)": np.exp(3 * x) + 0.05 * rng.normal(size=n),
        "U (x^2)":       x ** 2 + 0.02 * rng.normal(size=n),
        "circulo":       np.sqrt(np.maximum(1 - x ** 2, 0)) * rng.choice([-1, 1], n),
        "seno":          np.sin(4 * np.pi * x) + 0.05 * rng.normal(size=n),
    }
    from scipy.stats import spearmanr, pearsonr
    print(f"{'caso':16} {'Pearson':>9} {'Spearman':>9} {'Hoeffding D':>12}")
    for k, y in casos.items():
        print(f"{k:16} {pearsonr(x,y)[0]:9.3f} {spearmanr(x,y)[0]:9.3f} {hoeffding_d(x,y):12.4f}")
