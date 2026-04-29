"""
summary_stats.py
----------------
Statistiques résumées pour ABC sur le modèle MA(2).

Pourquoi ces statistiques ?
  Un MA(2) gaussien est entièrement caractérisé par ses autocorrélations
  aux lags 1 et 2 (rho_1 et rho_2). Au-delà du lag 2, l'autocorrélation
  est nulle. Ce sont donc des statistiques résumées quasi-suffisantes.

Organisation :
  - Pour ABC standard  : on utilise le vecteur (rho_1, rho_2) de dimension 2
  - Pour ABC-Gibbs     : on utilise UNE stat par composante :
      * theta_1 → rho_1 seul (dimension 1)
      * theta_2 → rho_2 seul (dimension 1)

  C'est tout l'intérêt d'ABC-Gibbs : chaque étape travaille en dimension 1
  au lieu de dimension 2, ce qui rend l'acceptation beaucoup plus facile.
"""

import numpy as np


# ==========================================================
#  1. CALCUL DES AUTOCORRÉLATIONS
# ==========================================================

def autocorrelations(y, max_lag=2):
    """
    Calcule les autocorrélations empiriques aux lags 1, ..., max_lag.

    Formule :
      rho(k) = gamma(k) / gamma(0)
      gamma(k) = (1/T) * sum_{t=k+1}^{T} (y_t - y_bar)(y_{t-k} - y_bar)

    Paramètres
    ----------
    y       : ndarray de taille (T,), la série temporelle
    max_lag : int, lag maximal

    Retourne
    --------
    rho : ndarray de taille (max_lag,)
          rho[0] = autocorrélation au lag 1
          rho[1] = autocorrélation au lag 2
    """
    T = len(y)
    y_centered = y - y.mean()
    gamma0 = np.sum(y_centered ** 2) / T

    if gamma0 == 0:
        return np.zeros(max_lag)

    rho = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        gamma_k = np.sum(y_centered[k:] * y_centered[:-k]) / T
        rho[k - 1] = gamma_k / gamma0
    return rho


def autocorrelations_batch(Y, max_lag=2):
    """
    Version vectorisée : calcule les autocorrélations pour un paquet
    de séries d'un coup. Indispensable pour que ABC-rejet soit rapide.

    Paramètres
    ----------
    Y : ndarray de taille (n_sims, T)
        Chaque ligne est une série temporelle.

    Retourne
    --------
    rho : ndarray de taille (n_sims, max_lag)
    """
    n_sims, T = Y.shape
    Y_centered = Y - Y.mean(axis=1, keepdims=True)
    gamma0 = np.sum(Y_centered ** 2, axis=1) / T  # shape (n_sims,)

    rho = np.empty((n_sims, max_lag))
    for k in range(1, max_lag + 1):
        gamma_k = np.sum(Y_centered[:, k:] * Y_centered[:, :-k], axis=1) / T
        # Éviter la division par zéro
        rho[:, k - 1] = np.where(gamma0 > 0, gamma_k / gamma0, 0.0)
    return rho


# ==========================================================
#  2. STATISTIQUES POUR ABC STANDARD (dimension 2)
# ==========================================================

def summary_full(y):
    """
    Statistique résumée pour ABC standard : le vecteur (rho_1, rho_2).
    Dimension 2.
    """
    return autocorrelations(y, max_lag=2)


def summary_full_batch(Y):
    """
    Version batch de summary_full.
    Entrée : (n_sims, T) → Sortie : (n_sims, 2)
    """
    return autocorrelations_batch(Y, max_lag=2)


# ==========================================================
#  3. STATISTIQUES POUR ABC-GIBBS (dimension 1 chacune)
# ==========================================================

def summary_component_1(y):
    """
    Statistique résumée pour la mise à jour de theta_1 dans ABC-Gibbs.
    On utilise rho_1 seul → dimension 1.

    Retourne un ndarray de taille (1,) pour garder un format cohérent
    avec les fonctions de distance.
    """
    return autocorrelations(y, max_lag=1)  # shape (1,)


def summary_component_2(y):
    """
    Statistique résumée pour la mise à jour de theta_2 dans ABC-Gibbs.
    On utilise rho_2 seul → dimension 1.

    Retourne un ndarray de taille (1,).
    """
    rho = autocorrelations(y, max_lag=2)
    return rho[1:]  # shape (1,), juste rho_2


# ==========================================================
#  4. TEST RAPIDE
# ==========================================================

if __name__ == "__main__":
    import ma_model

    rng = np.random.default_rng(42)
    theta_true = (0.6, 0.2)
    T = 500

    # Simuler une série
    y = ma_model.simulate(theta_true, T, rng)

    # Autocorrélations empiriques
    rho = summary_full(y)
    print(f"Autocorrélations empiriques : rho_1={rho[0]:.4f}, rho_2={rho[1]:.4f}")

    # Valeurs théoriques pour comparaison
    # rho_1 = (theta_1 + theta_1*theta_2) / (1 + theta_1^2 + theta_2^2)
    # rho_2 = theta_2 / (1 + theta_1^2 + theta_2^2)
    th1, th2 = theta_true
    denom = 1 + th1 ** 2 + th2 ** 2
    rho1_theo = (th1 + th1 * th2) / denom
    rho2_theo = th2 / denom
    print(f"Autocorrélations théoriques : rho_1={rho1_theo:.4f}, rho_2={rho2_theo:.4f}")
    print(f"(Les valeurs empiriques doivent être proches des théoriques pour T grand)")

    # Test batch
    Y = ma_model.simulate_batch(theta_true, T, n_sims=1000, rng=rng)
    rho_batch = summary_full_batch(Y)
    print(f"\nBatch de 1000 séries :")
    print(f"  rho_1 moyen = {rho_batch[:, 0].mean():.4f} (théo: {rho1_theo:.4f})")
    print(f"  rho_2 moyen = {rho_batch[:, 1].mean():.4f} (théo: {rho2_theo:.4f})")

    # Test des stats par composante
    s1 = summary_component_1(y)
    s2 = summary_component_2(y)
    print(f"\nStats par composante (pour ABC-Gibbs) :")
    print(f"  s1 (rho_1) = {s1}, shape = {s1.shape}")
    print(f"  s2 (rho_2) = {s2}, shape = {s2.shape}")
    