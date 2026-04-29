"""
samplers/base.py
----------------
Structure commune pour les résultats de tous les samplers.

Chaque algorithme (RWMH, ABC-rejet, ABC-Gibbs) renvoie un objet
SamplerResult, ce qui permet de les comparer de manière uniforme
dans evaluation.py et plots.py.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SamplerResult:
    """
    Résultat d'un sampler.

    Attributs
    ---------
    samples : ndarray de taille (N, 2)
        Échantillons de theta = (theta_1, theta_2).
        Pour RWMH : échantillons de la vraie postérieure.
        Pour ABC  : échantillons de la postérieure approchée.

    cpu_time : float
        Temps CPU total en secondes.

    n_model_calls : int
        Nombre total d'appels au modèle :
          - Pour RWMH     : nombre d'évaluations de la vraisemblance
          - Pour ABC-rejet : nombre de simulations de données
          - Pour ABC-Gibbs : nombre de simulations de données
        C'est la métrique clé pour comparer les coûts à budget égal.

    diagnostics : dict
        Informations supplémentaires propres à chaque algorithme
        (taux d'acceptation, ESS, epsilon, chaîne complète, etc.).
    """
    samples: np.ndarray
    cpu_time: float
    n_model_calls: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_samples(self):
        """Nombre d'échantillons (après burn-in pour MCMC)."""
        return self.samples.shape[0]

    @property
    def mean(self):
        """Moyenne postérieure estimée."""
        return self.samples.mean(axis=0)

    @property
    def var(self):
        """Variance postérieure estimée."""
        return self.samples.var(axis=0)

    def quantiles(self, q=(0.025, 0.5, 0.975)):
        """Quantiles par composante."""
        return np.quantile(self.samples, q, axis=0)

    def summary(self):
        """Affiche un résumé rapide."""
        print(f"  Échantillons   : {self.n_samples}")
        print(f"  Temps CPU      : {self.cpu_time:.2f}s")
        print(f"  Appels modèle  : {self.n_model_calls}")
        print(f"  Moyenne        : theta_1={self.mean[0]:.4f}, "
              f"theta_2={self.mean[1]:.4f}")
        print(f"  Écart-type     : theta_1={np.sqrt(self.var[0]):.4f}, "
              f"theta_2={np.sqrt(self.var[1]):.4f}")
        q = self.quantiles()
        print(f"  IC 95% theta_1 : [{q[0, 0]:.4f}, {q[2, 0]:.4f}]")
        print(f"  IC 95% theta_2 : [{q[0, 1]:.4f}, {q[2, 1]:.4f}]")
        for key, val in self.diagnostics.items():
            if key == "full_chain":
                continue  # trop gros pour afficher
            print(f"  {key:16s}: {val}")
        