"""
plots.py
--------
Fonctions de visualisation pour analyser et comparer les résultats des échantillonneurs.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
from samplers.base import SamplerResult

# Configuration pour avoir de jolis graphiques par défaut
sns.set_theme(style="whitegrid")

def plot_traces(result: SamplerResult, title: str = "Trace MCMC"):
    """
    Affiche l'évolution des échantillons (trace plot) au fil des itérations.
    Indispensable pour vérifier la convergence du RWMH et du ABC-Gibbs.
    """
    samples = result.samples
    n_iters = samples.shape[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(range(n_iters), samples[:, 0], color='blue', alpha=0.7)
    axes[0].set_ylabel(r"$\theta_1$")
    axes[0].set_title(title)
    
    axes[1].plot(range(n_iters), samples[:, 1], color='orange', alpha=0.7)
    axes[1].set_ylabel(r"$\theta_2$")
    axes[1].set_xlabel("Itération")
    
    plt.tight_layout()
    plt.show()

def plot_marginals(results_dict: Dict[str, SamplerResult], title: str = "Distributions Marginales Postérieures"):
    """
    Superpose les densités (KDE) de plusieurs algorithmes pour comparer visuellement l'erreur inférentielle.
    
    Paramètres
    ----------
    results_dict : Dict[str, SamplerResult]
        Un dictionnaire du type {"RWMH (Gold)": res_gold, "ABC-Reject": res_rej, "ABC-Gibbs": res_gibbs}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['black', 'blue', 'red'] # Noir pour le gold standard, couleurs pour ABC
    
    for (name, result), color in zip(results_dict.items(), colors):
        samples = result.samples
        # Marginal de theta 1
        sns.kdeplot(samples[:, 0], ax=axes[0], label=name, color=color, fill=True, alpha=0.1)
        # Marginal de theta 2
        sns.kdeplot(samples[:, 1], ax=axes[1], label=name, color=color, fill=True, alpha=0.1)
        
    axes[0].set_title(r"Densité marginale de $\theta_1$")
    axes[0].set_xlabel(r"$\theta_1$")
    axes[0].set_ylabel("Densité")
    axes[0].legend()
    
    axes[1].set_title(r"Densité marginale de $\theta_2$")
    axes[1].set_xlabel(r"$\theta_2$")
    axes[1].set_ylabel("Densité")
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_monte_carlo_error_boxplots(replicas_dict: Dict[str, List[SamplerResult]]):
    """
    Génère des boxplots des moyennes postérieures sur plusieurs runs (répond à la consigne du PDF).
    Permet de visualiser l'erreur Monte Carlo de chaque algorithme.
    
    Paramètres
    ----------
    replicas_dict : Dict[str, List[SamplerResult]]
        Exemple : {"RWMH": [res1, res2...], "ABC-Reject": [res1...], "ABC-Gibbs": [res1...]}
    """
    data_theta1 = []
    data_theta2 = []
    labels = []
    
    # Extraction des moyennes de chaque run pour chaque algorithme
    for algo_name, replicas in replicas_dict.items():
        for res in replicas:
            labels.append(algo_name)
            data_theta1.append(res.mean[0])
            data_theta2.append(res.mean[1])
            
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot pour theta 1
    sns.boxplot(x=labels, y=data_theta1, ax=axes[0], palette="Set2")
    axes[0].set_title(r"Dispersion de l'estimateur de la moyenne ($\theta_1$)")
    axes[0].set_ylabel(r"Moyenne estimée de $\theta_1$")
    
    # Boxplot pour theta 2
    sns.boxplot(x=labels, y=data_theta2, ax=axes[1], palette="Set2")
    axes[1].set_title(r"Dispersion de l'estimateur de la moyenne ($\theta_2$)")
    axes[1].set_ylabel(r"Moyenne estimée de $\theta_2$")
    
    plt.suptitle("Évaluation de l'erreur Monte Carlo (Box-plots sur runs répétés)")
    plt.tight_layout()
    plt.show()