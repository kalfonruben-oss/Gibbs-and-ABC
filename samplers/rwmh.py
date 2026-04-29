"""
samplers/rwmh.py
----------------
Random Walk Metropolis-Hastings sur la VRAIE postérieure du MA(2).

C'est le « gold standard » du projet : il cible la postérieure exacte
pi(theta | y) ∝ prior(theta) * L(theta ; y)
où L est la vraisemblance gaussienne exacte (calculée dans ma_model.py).

Algorithme :
  1. Proposer theta' = theta + scale * N(0, I_2)
  2. Calculer alpha = pi(theta' | y) / pi(theta | y)
  3. Accepter theta' avec probabilité min(1, alpha)

Règle de calibration : viser un taux d'acceptation autour de 25-35%
en ajustant proposal_scale. Trop haut → la chaîne bouge peu à chaque
pas. Trop bas → la chaîne rejette trop et reste bloquée.
"""

import time
import numpy as np
from tqdm import trange

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .base import SamplerResult
import ma_model


# ==========================================================
#  1. L'ALGORITHME PRINCIPAL
# ==========================================================

def run(y_obs, n_iter=50_000, proposal_scale=0.1, theta_init=None,
        burnin=5000, rng=None, sigma=ma_model.SIGMA, show_progress=True):
    """
    Random Walk Metropolis-Hastings.

    Paramètres
    ----------
    y_obs          : ndarray (T,), les données observées
    n_iter         : int, nombre total d'itérations (burn-in inclus)
    proposal_scale : float, écart-type de la gaussienne de proposition.
                     Commencez par 0.1, puis ajustez pour avoir ~30% d'acceptation.
    theta_init     : (2,) ou None. Si None, tiré du prior.
    burnin         : int, itérations de burn-in à jeter au début.
    rng            : np.random.Generator
    sigma          : float, écart-type du bruit (fixé à 1)
    show_progress  : bool, afficher la barre de progression

    Retourne
    --------
    SamplerResult
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialisation
    if theta_init is None:
        theta = ma_model.sample_prior(rng, n=1)[0]
    else:
        theta = np.array(theta_init, dtype=float)

    log_post_current = ma_model.log_posterior(theta, y_obs, sigma)

    # Stockage
    chain = np.empty((n_iter, 2))
    n_accept = 0
    n_model_calls = 0

    t0 = time.process_time()
    iterator = trange(n_iter, desc="RWMH", disable=not show_progress)

    for i in iterator:
        # --- Étape 1 : proposer ---
        theta_prop = theta + proposal_scale * rng.standard_normal(2)

        # --- Étape 2 : évaluer la log-postérieure ---
        log_post_prop = ma_model.log_posterior(theta_prop, y_obs, sigma)
        n_model_calls += 1

        # --- Étape 3 : accepter/rejeter ---
        log_alpha = log_post_prop - log_post_current
        if np.log(rng.uniform()) < log_alpha:
            theta = theta_prop
            log_post_current = log_post_prop
            n_accept += 1

        chain[i] = theta

        # Afficher le taux d'acceptation en cours
        if (i + 1) % 5000 == 0:
            rate = n_accept / (i + 1)
            iterator.set_postfix(accept=f"{rate:.1%}")

    cpu_time = time.process_time() - t0
    accept_rate = n_accept / n_iter

    # Retirer le burn-in
    samples = chain[burnin:]

    return SamplerResult(
        samples=samples,
        cpu_time=cpu_time,
        n_model_calls=n_model_calls,
        diagnostics={
            "accept_rate": accept_rate,
            "burnin": burnin,
            "proposal_scale": proposal_scale,
            "full_chain": chain,  # gardé pour les trace plots
        }
    )


# ==========================================================
#  2. EFFECTIVE SAMPLE SIZE (ESS)
# ==========================================================

def effective_sample_size(chain):
    """
    Calcule l'ESS (Effective Sample Size) par composante.

    L'ESS mesure combien d'échantillons indépendants « valent »
    vos N échantillons corrélés de MCMC.
    ESS = N / (1 + 2 * sum des autocorrélations)

    On utilise la méthode de Geyer (initial positive sequence) :
    on s'arrête quand la somme de 2 autocorrélations consécutives
    devient négative.

    Paramètres
    ----------
    chain : ndarray (N, 2), la chaîne APRÈS burn-in

    Retourne
    --------
    ess : ndarray (2,), l'ESS pour theta_1 et theta_2
    """
    N, d = chain.shape
    ess = np.empty(d)

    for j in range(d):
        x = chain[:, j] - chain[:, j].mean()

        # Autocorrélation via FFT (rapide)
        fft_x = np.fft.fft(x, n=2 * N)
        acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:N]
        acf = acf / acf[0]

        # Troncature de Geyer : on s'arrête quand la somme de
        # deux autocorrélations consécutives devient négative
        total = 0.0
        for k in range(1, N // 2):
            pair_sum = acf[2 * k - 1] + acf[2 * k]
            if pair_sum < 0:
                break
            total += pair_sum

        ess[j] = N / (1.0 + 2.0 * total)

    return ess


# ==========================================================
#  3. TEST RAPIDE
# ==========================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Générer des données
    theta_true = (0.6, 0.2)
    T = 100  # T=100 pour que le test soit rapide (~1s)
    y_obs = ma_model.simulate(theta_true, T, rng)

    print(f"Données : T={T}, theta_vrai={theta_true}")
    print(f"{'='*50}")

    # Lancer RWMH (3000 itérations pour le test, mettre 50_000+ pour le vrai run)
    result = run(y_obs, n_iter=3000, proposal_scale=0.15,
                 burnin=500, rng=rng, show_progress=False)

    print(f"\nRésultats RWMH :")
    result.summary()

    # ESS
    ess = effective_sample_size(result.samples)
    print(f"\n  ESS            : theta_1={ess[0]:.0f}, theta_2={ess[1]:.0f}")
    print(f"  ESS/seconde    : theta_1={ess[0]/result.cpu_time:.0f}, "
          f"theta_2={ess[1]/result.cpu_time:.0f}")

    # Conseil de calibration
    ar = result.diagnostics["accept_rate"]
    if ar > 0.4:
        print(f"\n  ⚠ Taux d'acceptation = {ar:.0%} (trop haut)")
        print(f"    → Augmentez proposal_scale pour explorer plus vite")
    elif ar < 0.15:
        print(f"\n  ⚠ Taux d'acceptation = {ar:.0%} (trop bas)")
        print(f"    → Diminuez proposal_scale")
    else:
        print(f"\n  ✓ Taux d'acceptation = {ar:.0%} (bon, visez 25-35%)")