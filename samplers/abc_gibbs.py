"""
samplers/abc_gibbs.py
---------------------
Implémentation de l'algorithme ABC-Gibbs (inspiré de Clarté et al. 2019).
Met à jour les paramètres composante par composante (coordonnée par coordonnée).
"""

import time
import numpy as np
from base import SamplerResult

class ABCGibbsSampler:
    """
    Échantillonneur ABC-Gibbs.
    Utilise une approche de type rejet (ou MCMC) conditionnel pour chaque coordonnée.
    """
    
    def __init__(self, conditional_prior_sampler, simulator, summary_stats, distance_metric):
        """
        Initialise l'ABC-Gibbs.
        
        Paramètres
        ----------
        conditional_prior_sampler : callable
            Fonction `f(j, theta_actuel)` qui renvoie un échantillon pour la 
            composante `j` sachant les autres composantes de `theta_actuel`.
        simulator : callable
            Fonction qui prend theta en entrée et retourne des données simulées.
        summary_stats : callable
            Fonction qui calcule les statistiques résumées des données.
        distance_metric : callable
            Fonction qui calcule la distance entre deux jeux de stats résumées.
        """
        self.conditional_prior_sampler = conditional_prior_sampler
        self.simulator = simulator
        self.summary_stats = summary_stats
        self.distance_metric = distance_metric

    def sample(self, y_obs, theta_init, n_samples: int, epsilon: float, max_calls_per_step: int = 10_000) -> SamplerResult:
        """
        Exécute l'algorithme ABC-Gibbs.
        
        Paramètres
        ----------
        y_obs : ndarray
            Les données observées.
        theta_init : array-like
            Le point de départ de la chaîne (ex: [0.0, 0.0]).
        n_samples : int
            Nombre d'itérations de Gibbs souhaitées (longueur de la chaîne).
        epsilon : float
            Seuil de tolérance ABC.
        max_calls_per_step : int
            Sécurité : nombre max d'essais pour mettre à jour UNE coordonnée. 
            Évite une boucle infinie si on est bloqué.
            
        Retours
        -------
        SamplerResult
        """
        s_obs = self.summary_stats(y_obs)
        
        # Initialisation
        dim = len(theta_init)
        samples = np.zeros((n_samples, dim))
        samples[0] = np.copy(theta_init)
        
        n_calls = 0
        stuck_warnings = 0
        
        start_time = time.process_time()
        
        # Boucle principale de Gibbs
        for i in range(1, n_samples):
            theta_curr = np.copy(samples[i-1])
            
            # Mise à jour composante par composante (j = 0, puis j = 1)
            for j in range(dim):
                accepted = False
                calls_for_this_step = 0
                
                # On cherche un candidat acceptable pour la coordonnée j
                while not accepted and calls_for_this_step < max_calls_per_step:
                    # 1. Tirer un candidat pour la coordonnée j (conditionnellement aux autres)
                    theta_star_j = self.conditional_prior_sampler(j, theta_curr)
                    
                    # 2. Créer le vecteur theta candidat
                    theta_cand = np.copy(theta_curr)
                    theta_cand[j] = theta_star_j
                    
                    # 3. Simuler et évaluer
                    y_sim = self.simulator(theta_cand)
                    n_calls += 1
                    calls_for_this_step += 1
                    
                    s_sim = self.summary_stats(y_sim)
                    d = self.distance_metric(s_sim, s_obs)
                    
                    # 4. Acceptation ABC
                    if d <= epsilon:
                        theta_curr[j] = theta_star_j
                        accepted = True
                
                # Si on a atteint la limite d'appels sans trouver de point, 
                # on garde l'ancienne valeur pour cette coordonnée (comportement MCMC)
                if not accepted:
                    stuck_warnings += 1
                    
            # Fin de l'itération i, on stocke le vecteur mis à jour
            samples[i] = np.copy(theta_curr)
            
        cpu_time = time.process_time() - start_time
        
        diagnostics = {
            "algorithm": "ABC-Gibbs",
            "epsilon": epsilon,
            "stuck_warnings": stuck_warnings,
            "avg_calls_per_iteration": n_calls / n_samples
        }
        
        return SamplerResult(
            samples=samples,
            cpu_time=cpu_time,
            n_model_calls=n_calls,
            diagnostics=diagnostics
        )