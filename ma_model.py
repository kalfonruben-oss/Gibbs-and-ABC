"""
ma_model.py
-----------
Modèle MA(2) : y_t = z_t + theta_1 * z_{t-1} + theta_2 * z_{t-2}
avec z_t ~ N(0, sigma^2) i.i.d.

On fixe sigma = 1 (connu) par simplicité.
Ce fichier contient :
  - simulate()          : simule une série MA(2)
  - simulate_batch()    : simule plusieurs séries d'un coup (vectorisé)
  - log_likelihood()    : vraisemblance exacte via Cholesky
  - log_prior()         : prior uniforme sur le triangle d'inversibilité
  - log_posterior()     : log-postérieure = log_prior + log_likelihood
  - sample_prior()      : tire des échantillons du prior
"""

import numpy as np
from scipy.linalg import toeplitz, cho_factor, cho_solve


SIGMA = 1.0  # écart-type du bruit, fixé et connu


# ==========================================================
#  1. SIMULATION
# ==========================================================

def simulate(theta, T, rng, sigma=SIGMA):
    """
    Simule T observations d'un MA(2).

    Paramètres
    ----------
    theta : tuple (theta_1, theta_2)
    T     : int, longueur de la série
    rng   : np.random.Generator
    sigma : float, écart-type du bruit

    Retourne
    --------
    y : ndarray de taille (T,)
    """
    th1, th2 = theta
    # On génère T+2 bruits car y_1 utilise z_0 et z_{-1}
    z = rng.normal(0, sigma, size=T + 2)
    y = np.empty(T)
    for t in range(T):
        y[t] = z[t + 2] + th1 * z[t + 1] + th2 * z[t]
    return y


def simulate_batch(theta, T, n_sims, rng, sigma=SIGMA):
    """
    Simule n_sims séries de longueur T d'un coup (vectorisé, beaucoup
    plus rapide que d'appeler simulate() dans une boucle).

    Retourne
    --------
    Y : ndarray de taille (n_sims, T)
    """
    th1, th2 = theta
    Z = rng.normal(0, sigma, size=(n_sims, T + 2))
    Y = Z[:, 2:] + th1 * Z[:, 1:-1] + th2 * Z[:, :-2]
    return Y


# ==========================================================
#  2. VRAISEMBLANCE EXACTE
# ==========================================================

def autocovariances(theta, sigma=SIGMA):
    """
    Renvoie les autocovariances gamma_0, gamma_1, gamma_2 du MA(2).

    Formules :
      gamma_0 = sigma^2 * (1 + theta_1^2 + theta_2^2)
      gamma_1 = sigma^2 * (theta_1 + theta_1 * theta_2)
      gamma_2 = sigma^2 * theta_2
      gamma_h = 0  pour h >= 3
    """
    th1, th2 = theta
    v = sigma ** 2
    gamma0 = v * (1.0 + th1 ** 2 + th2 ** 2)
    gamma1 = v * (th1 + th1 * th2)
    gamma2 = v * th2
    return gamma0, gamma1, gamma2


def log_likelihood(theta, y, sigma=SIGMA):
    """
    Log-vraisemblance exacte du MA(2).

    Le vecteur y = (y_1, ..., y_T) suit une loi N(0, Sigma) où Sigma
    est une matrice Toeplitz construite à partir des autocovariances.
    On factorise Sigma par Cholesky pour calculer :
      log L = -T/2 log(2pi) - 1/2 log|Sigma| - 1/2 y^T Sigma^{-1} y

    Complexité : O(T^3) en force brute. Suffisant pour T <= 1000.

    Paramètres
    ----------
    theta : tuple (theta_1, theta_2)
    y     : ndarray de taille (T,)
    sigma : float

    Retourne
    --------
    ll : float, la log-vraisemblance
    """
    T = len(y)
    g0, g1, g2 = autocovariances(theta, sigma)

    # Construire la première ligne de la matrice Toeplitz
    first_row = np.zeros(T)
    first_row[0] = g0
    if T > 1:
        first_row[1] = g1
    if T > 2:
        first_row[2] = g2
    # Les autres entrées restent à 0 (gamma_h = 0 pour h >= 3)

    Sigma = toeplitz(first_row)

    # Factorisation de Cholesky
    try:
        c, low = cho_factor(Sigma)
    except np.linalg.LinAlgError:
        return -np.inf  # matrice non définie positive

    # log|Sigma| = 2 * somme des log des éléments diagonaux de L
    log_det = 2.0 * np.sum(np.log(np.diag(c)))

    # y^T Sigma^{-1} y
    alpha = cho_solve((c, low), y)
    quad = y @ alpha

    ll = -0.5 * (T * np.log(2.0 * np.pi) + log_det + quad)
    return ll


# ==========================================================
#  3. PRIOR : uniforme sur le triangle d'inversibilité
# ==========================================================

def in_invertibility_region(theta):
    """
    Vérifie si theta est dans la région d'inversibilité du MA(2).

    Les conditions sont :
      |theta_2| < 1
      theta_1 + theta_2 > -1
      theta_2 - theta_1 > -1

    Géométriquement, c'est un triangle dans le plan (theta_1, theta_2).
    """
    th1, th2 = theta
    return (abs(th2) < 1) and (th1 + th2 > -1) and (th2 - th1 > -1)


def log_prior(theta):
    """
    Log du prior : uniforme sur le triangle d'inversibilité.
    Renvoie 0 si theta est dedans, -inf sinon.
    (La constante de normalisation n'est pas nécessaire pour le MCMC.)
    """
    if in_invertibility_region(theta):
        return 0.0
    return -np.inf


def log_posterior(theta, y, sigma=SIGMA):
    """
    Log-postérieure non normalisée :
      log pi(theta | y) = log prior(theta) + log L(theta ; y)
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, sigma)


def sample_prior(rng, n=1):
    """
    Tire n échantillons uniformes dans le triangle d'inversibilité
    par méthode de rejet.

    On tire (theta_1, theta_2) uniformément dans [-2,2] x [-1,1]
    et on garde seulement ceux qui tombent dans le triangle.
    Le taux d'acceptation est environ 50%.

    Retourne
    --------
    samples : ndarray de taille (n, 2)
    """
    samples = []
    while len(samples) < n:
        th1 = rng.uniform(-2, 2)
        th2 = rng.uniform(-1, 1)
        if in_invertibility_region((th1, th2)):
            samples.append([th1, th2])
    return np.array(samples)


# ==========================================================
#  4. PETIT TEST RAPIDE
# ==========================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Paramètres vrais
    theta_true = (0.6, 0.2)
    T = 200

    # Simuler des données
    y = simulate(theta_true, T, rng)
    print(f"Données simulées : T={T}, mean={y.mean():.3f}, std={y.std():.3f}")

    # Évaluer la vraisemblance au vrai theta et à un faux
    ll_true = log_likelihood(theta_true, y)
    ll_faux = log_likelihood((0.0, 0.0), y)
    print(f"Log-likelihood au vrai theta : {ll_true:.2f}")
    print(f"Log-likelihood à theta=(0,0) : {ll_faux:.2f}")
    print(f"Différence : {ll_true - ll_faux:.2f} (doit être > 0)")

    # Vérifier le prior
    samples = sample_prior(rng, n=5)
    print(f"\n5 échantillons du prior :")
    for s in samples:
        print(f"  theta=({s[0]:.3f}, {s[1]:.3f}), "
              f"dans le triangle: {in_invertibility_region(s)}")