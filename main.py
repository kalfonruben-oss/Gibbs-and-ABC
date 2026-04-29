import numpy as np
import matplotlib.pyplot as plt
from samplers.rwmh import run as run_rwmh
from samplers.abc_reject import ABCRejectSampler
from samplers.abc_gibbs import ABCGibbsSampler
from evaluations import compute_inferential_error, run_replicas, compute_monte_carlo_error
import plots

# --- DUMMY MA(2) MODEL POUR LE TEST (en attendant le code de ton binôme) ---
def simulate_ma2(theta, n=100):
    t1, t2 = theta
    errors = np.random.normal(0, 1, n + 2)
    y = errors[2:] + t1 * errors[1:-1] + t2 * errors[:-2]
    return y

def summary_stats_ma2(y):
    # Autocovariances d'ordre 1 et 2
    return np.array([np.cov(y[1:], y[:-1])[0,1], np.cov(y[2:], y[:-2])[0,1]])

def log_prior_ma2(theta):
    t1, t2 = theta
    if (t2 < 1) and (t2 + t1 > -1) and (t2 - t1 > -1):
        return 0.0 # Uniforme sur le triangle
    return -np.inf

def log_likelihood_ma2(y, theta):
    # Approximation Gaussienne simple pour le test
    y_sim = simulate_ma2(theta, len(y))
    return -0.5 * np.sum((y - y_sim)**2)

def sample_conditional_prior(j, theta):
    t1, t2 = theta
    if j == 0: return np.random.uniform(-1 - t2, 1 + t2)
    else: return np.random.uniform(abs(t1) - 1, 1)

# --- WORKFLOW DE TEST ---
def main():
    print("--- Démarrage du Test Global ---")
    theta_true = np.array([0.6, 0.2])
    y_obs = simulate_ma2(theta_true)
    epsilon = 0.5
    
    # 1. Gold Standard (RWMH)
    print("\nLancement RWMH...")
    res_gold = run_rwmh(y_obs, n_iter=2000, proposal_scale=0.1, theta_init=[0.,0.],burnin=500)

    # 2. ABC Rejet
    print("Lancement ABC-Rejet...")
    def prior_sampler():
        while True:
            t = np.random.uniform(-2, 2, 2)
            if log_prior_ma2(t) == 0: return t
            
    abc_rej = ABCRejectSampler(prior_sampler, simulate_ma2, summary_stats_ma2, lambda a,b: np.linalg.norm(a-b))
    res_rej = abc_rej.sample(y_obs, 500, epsilon)

    # 3. ABC Gibbs
    print("Lancement ABC-Gibbs...")
    abc_gibbs = ABCGibbsSampler(sample_conditional_prior, simulate_ma2, summary_stats_ma2, lambda a,b: np.linalg.norm(a-b))
    res_gibbs = abc_gibbs.sample(y_obs, np.array([0.,0.]), 500, epsilon)

    # --- Résultats ---
    res_gold.summary()
    res_rej.summary()
    res_gibbs.summary()

    # --- Plots ---
    plots.plot_marginals({
        "Gold Standard (RWMH)": res_gold,
        "ABC-Reject": res_rej,
        "ABC-Gibbs": res_gibbs
    })

if __name__ == "__main__":
    main()