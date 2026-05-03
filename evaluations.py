"""
evaluations.py
-------------
Fonctions pour évaluer et comparer les performances des différents échantillonneurs,
comme demandé dans les consignes : temps CPU, erreur inférentielle, erreur Monte Carlo.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import List
from samplers.base import SamplerResult

def compute_inferential_error(abc_result: SamplerResult, gold_result: SamplerResult) -> dict:
    """
    Calcule l'erreur inférentielle entre une méthode ABC et le Gold Standard (RWMH).
    On utilise la distance de Wasserstein (Earth Mover's Distance) marginale pour chaque paramètre.
    
    Paramètres
    ----------
    abc_result : SamplerResult
        Résultat de l'algorithme ABC (Rejet ou Gibbs).
    gold_result : SamplerResult
        Résultat de l'algorithme RWMH (la vraie postérieure).
        
    Retours
    -------
    dict
        Distance pour theta_1, theta_2, et la distance moyenne.
    """
    samples_abc = abc_result.samples
    samples_gold = gold_result.samples
    
    # Distance de Wasserstein pour la première composante (theta 1)
    w_dist_1 = wasserstein_distance(samples_abc[:, 0], samples_gold[:, 0])
    
    # Distance de Wasserstein pour la deuxième composante (theta 2)
    w_dist_2 = wasserstein_distance(samples_abc[:, 1], samples_gold[:, 1])
    
    return {
        "wasserstein_theta1": w_dist_1,
        "wasserstein_theta2": w_dist_2,
        "wasserstein_mean": (w_dist_1 + w_dist_2) / 2
    }

def compute_monte_carlo_error(replica_results: List[SamplerResult]) -> dict:
    """
    Calcule l'erreur Monte Carlo à partir de plusieurs exécutions (replicas) du même algorithme.
    L'erreur MC est estimée par la variance empirique des moyennes postérieures.
    
    Paramètres
    ----------
    replica_results : List[SamplerResult]
        Une liste de résultats obtenus en lançant le même algorithme plusieurs fois 
        avec des graines aléatoires (seeds) différentes.
        
    Retours
    -------
    dict
        Variance de l'estimateur de la moyenne pour theta_1 et theta_2.
    """
    if len(replica_results) < 2:
        raise ValueError("Il faut au moins 2 replicas pour calculer une variance (Erreur Monte Carlo).")
        
    # Extraire les moyennes postérieures de chaque run
    # Chaque 'mean' est un array [mean_theta1, mean_theta2]
    means = np.array([res.mean for res in replica_results])
    
    # Calculer la variance de ces moyennes sur les N runs
    mc_error_theta1 = np.var(means[:, 0], ddof=1)
    mc_error_theta2 = np.var(means[:, 1], ddof=1)
    
    # On peut aussi calculer le temps CPU moyen sur ces runs
    avg_cpu_time = np.mean([res.cpu_time for res in replica_results])
    
    return {
        "mc_error_theta1": mc_error_theta1,
        "mc_error_theta2": mc_error_theta2,
        "avg_cpu_time": avg_cpu_time,
        "n_replicas": len(replica_results)
    }

def run_replicas(sampler, sampler_kwargs: dict, n_replicas: int = 10) -> List[SamplerResult]:
    """
    Fonction utilitaire pour lancer un algorithme plusieurs fois (utile pour le notebook de comparaison).
    
    Paramètres
    ----------
    sampler : Objet Sampler (ex: RWMHSampler, ABCRejectSampler)
        L'échantillonneur à utiliser.
    sampler_kwargs : dict
        Les arguments à passer à la méthode `sample` (y_obs, n_samples, epsilon, etc.).
    n_replicas : int
        Nombre de fois où l'algorithme doit être relancé.
        
    Retours
    -------
    List[SamplerResult]
    """
    results = []
    print(f"Lancement de {n_replicas} replicas...")
    
    for i in range(n_replicas):
        # On peut fixer une seed différente pour chaque replica si on veut être strict, 
        # mais numpy s'en charge très bien tout seul si on ne la fixe pas.
        res = sampler.sample(**sampler_kwargs)
        results.append(res)
        print(f"  Replica {i+1}/{n_replicas} terminé en {res.cpu_time:.2f}s")
        
    return results