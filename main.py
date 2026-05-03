"""
main.py
-------
Point d'entrée principal du projet.
Lance les trois algorithmes (RWMH, ABC-Rejet, ABC-Gibbs) sur un jeu de données
MA(2) simulé, affiche les résumés statistiques et génère les graphiques.
"""

import numpy as np

import ma_model
import summary_stats
import distances
import plots
from samplers import run_rwmh, ABCRejectSampler, ABCGibbsSampler


def sample_conditional_prior(j, theta):
    """Loi uniforme conditionnelle respectant le triangle d'inversibilité du MA(2)."""
    t1, t2 = theta
    if j == 0:
        return np.random.uniform(-1.0 - t2, 1.0 + t2)
    else:
        return np.random.uniform(abs(t1) - 1.0, 1.0)


def main():
    print("=" * 55)
    print("  Projet ABC - Comparaison RWMH / ABC-Rejet / ABC-Gibbs")
    print("=" * 55)

    rng = np.random.default_rng(42)
    theta_true = np.array([0.6, 0.2])
    T = 200
    y_obs = ma_model.simulate(theta_true, T, rng)
    print(f"\nDonnees : T={T}, theta_vrai={theta_true}")

    epsilon = 0.5

    print("\n[1/3] Lancement RWMH (gold standard)...")
    res_gold = run_rwmh(y_obs, n_iter=5000, proposal_scale=0.15,
                        theta_init=[0.0, 0.0], burnin=500, rng=rng)
    res_gold.summary()

    print("\n[2/3] Lancement ABC-Rejet...")
    def prior_sampler():
        return ma_model.sample_prior(rng, n=1)[0]

    abc_rej = ABCRejectSampler(
        prior_sampler=prior_sampler,
        simulator=lambda theta: ma_model.simulate(theta, T, rng),
        summary_stats=summary_stats.summary_full,
        distance_metric=distances.euclidean,
    )
    res_rej = abc_rej.sample(y_obs, n_samples=500, epsilon=epsilon)
    res_rej.summary()

    print("\n[3/3] Lancement ABC-Gibbs...")
    abc_gibbs = ABCGibbsSampler(
        conditional_prior_sampler=sample_conditional_prior,
        simulator=lambda theta: ma_model.simulate(theta, T, rng),
        summary_stats=summary_stats.summary_full,
        distance_metric=distances.euclidean,
    )
    res_gibbs = abc_gibbs.sample(y_obs, theta_init=np.array([0.0, 0.0]),
                                 n_samples=500, epsilon=epsilon)
    res_gibbs.summary()

    print("\nGeneration des graphiques...")
    plots.plot_marginals({
        "Gold Standard (RWMH)": res_gold,
        "ABC-Rejet": res_rej,
        "ABC-Gibbs": res_gibbs,
    })


if __name__ == "__main__":
    main()
