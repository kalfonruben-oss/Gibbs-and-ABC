"""
distances.py
------------
Fonctions de distance entre statistiques résumées pour ABC.

Dans ABC, on accepte theta* si d(s(y_sim), s(y_obs)) < epsilon.
Ce fichier définit les distances d() disponibles.

Pour notre MA(2) :
  - ABC standard  : s est de dimension 2 → distance euclidienne dans R^2
  - ABC-Gibbs     : s est de dimension 1 → distance = simple valeur absolue
"""

import numpy as np


# ==========================================================
#  1. DISTANCE EUCLIDIENNE
# ==========================================================

def euclidean(s1, s2):
    """
    Distance euclidienne entre deux vecteurs de statistiques résumées.

    Marche en toute dimension (y compris dimension 1, où ça revient
    à la valeur absolue).

    Paramètres
    ----------
    s1, s2 : ndarray de même taille

    Retourne
    --------
    d : float
    """
    return np.sqrt(np.sum((s1 - s2) ** 2))


def euclidean_batch(S_sim, s_obs):
    """
    Version vectorisée : calcule la distance euclidienne entre chaque
    ligne de S_sim et le vecteur s_obs.

    Indispensable pour ABC-rejet rapide.

    Paramètres
    ----------
    S_sim : ndarray de taille (n_sims, d)
            Statistiques résumées de chaque simulation.
    s_obs : ndarray de taille (d,)
            Statistiques résumées des données observées.

    Retourne
    --------
    distances : ndarray de taille (n_sims,)
    """
    return np.sqrt(np.sum((S_sim - s_obs[None, :]) ** 2, axis=1))


# ==========================================================
#  2. DISTANCE L1 (Manhattan)
# ==========================================================

def absolute(s1, s2):
    """
    Distance L1 (somme des valeurs absolues).
    Peut être préférable quand les composantes de s ont des échelles
    très différentes et qu'on ne veut pas qu'une domine l'autre.
    """
    return np.sum(np.abs(s1 - s2))


def absolute_batch(S_sim, s_obs):
    """Version vectorisée de la distance L1."""
    return np.sum(np.abs(S_sim - s_obs[None, :]), axis=1)


# ==========================================================
#  3. TEST RAPIDE
# ==========================================================

if __name__ == "__main__":
    # Test simple
    s1 = np.array([0.5, 0.15])
    s2 = np.array([0.52, 0.13])
    print(f"s_obs = {s1}")
    print(f"s_sim = {s2}")
    print(f"Distance euclidienne : {euclidean(s1, s2):.6f}")
    print(f"Distance L1          : {absolute(s1, s2):.6f}")

    # Test batch
    S_batch = np.array([
        [0.52, 0.13],
        [0.10, 0.80],
        [0.49, 0.16],
    ])
    dists = euclidean_batch(S_batch, s1)
    print(f"\nDistances batch (3 simulations vs s_obs) :")
    for i, d in enumerate(dists):
        print(f"  sim {i+1} : s={S_batch[i]}, dist={d:.6f}")

    # Avec epsilon = 0.05, qui est accepté ?
    eps = 0.05
    mask = dists < eps
    print(f"\nAvec epsilon={eps}, acceptés : {np.where(mask)[0] + 1}")
