"""
samplers/abc_reject.py
----------------------
Implémentation de l'algorithme ABC par rejet standard.
"""

import time
import numpy as np
from .base import SamplerResult

class ABCRejectSampler:
    """
    Échantillonneur ABC par rejet (ABC-Reject).
    """
    
    def __init__(self, prior_sampler, simulator, summary_stats, distance_metric):
        """
        Initialise le sampler avec les composantes du modèle.
        
        Paramètres
        ----------
        prior_sampler : callable
            Fonction sans argument qui retourne un échantillon du prior (ex: theta = (theta_1, theta_2)).
        simulator : callable
            Fonction qui prend theta en entrée et retourne des données simulées.
        summary_stats : callable
            Fonction qui prend des données (observées ou simulées) et retourne les statistiques résumées.
        distance_metric : callable
            Fonction qui prend deux jeux de stats résumées et retourne une distance scalaire.
        """
        self.prior_sampler = prior_sampler
        self.simulator = simulator
        self.summary_stats = summary_stats
        self.distance_metric = distance_metric

    def sample(self, y_obs, n_samples: int, epsilon: float, max_calls: int = 10_000_000) -> SamplerResult:
        """
        Exécute l'algorithme ABC-rejet.
        
        Paramètres
        ----------
        y_obs : ndarray
            Les données observées.
        n_samples : int
            Le nombre d'échantillons acceptés désirés.
        epsilon : float
            Le seuil de tolérance pour la distance.
        max_calls : int
            Sécurité : nombre maximum d'appels au simulateur pour éviter une boucle infinie.
            
        Retours
        -------
        SamplerResult
            L'objet contenant les échantillons, le temps CPU, le nombre d'appels et les diagnostics.
        """
        # Calcul des statistiques résumées pour les données observées (une seule fois)
        s_obs = self.summary_stats(y_obs)
        
        accepted_samples = []
        accepted_distances = []
        n_calls = 0
        
        # Démarrage du chronomètre (process_time est idéal pour le temps CPU)
        start_time = time.process_time()
        
        while len(accepted_samples) < n_samples and n_calls < max_calls:
            # 1. Échantillonner theta depuis la distribution a priori
            theta_star = self.prior_sampler()
            
            # 2. Simuler des données pseudo-observées avec ce theta
            y_sim = self.simulator(theta_star)
            n_calls += 1
            
            # 3. Calculer les statistiques résumées des données simulées
            s_sim = self.summary_stats(y_sim)
            
            # 4. Calculer la distance entre les stats simulées et observées
            d = self.distance_metric(s_sim, s_obs)
            
            # 5. Accepter si la distance est inférieure ou égale au seuil epsilon
            if d <= epsilon:
                accepted_samples.append(theta_star)
                accepted_distances.append(d)
                
        cpu_time = time.process_time() - start_time
        
        # Avertissement si on a atteint la limite d'appels avant d'avoir le nombre d'échantillons voulu
        if len(accepted_samples) < n_samples:
            print(f"Attention: Arrêt prématuré après {max_calls} appels au modèle. "
                  f"Seulement {len(accepted_samples)} échantillons acceptés.")
            
        samples_array = np.array(accepted_samples)
        
        # Préparation des diagnostics
        acceptance_rate = len(accepted_samples) / n_calls if n_calls > 0 else 0.0
        
        diagnostics = {
            "algorithm": "ABC-Reject",
            "epsilon": epsilon,
            "acceptance_rate": acceptance_rate,
            "max_distance": np.max(accepted_distances) if accepted_distances else None,
            "mean_distance": np.mean(accepted_distances) if accepted_distances else None
        }
        
        return SamplerResult(
            samples=samples_array,
            cpu_time=cpu_time,
            n_model_calls=n_calls,
            diagnostics=diagnostics
        )