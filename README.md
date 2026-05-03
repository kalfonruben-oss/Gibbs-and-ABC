# 🎲 Overcoming the Curse of Dimensionality in ABC

> **A Comparative Study of Standard ABC and ABC-Gibbs on the MA(2) Model**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Welcome to the repository for our Bayesian Statistics project! This project explores the limitations of standard Approximate Bayesian Computation (ABC) in high-dimensional spaces and demonstrates how the **ABC-Gibbs** algorithm effectively breaks the curse of dimensionality using component-wise updates and local summary statistics.

---

## 📖 About the Project

Bayesian inference relies on evaluating the likelihood function, which is often intractable in complex real-world models. While **Approximate Bayesian Computation (ABC)** bypasses this by simulating data, standard ABC-Rejection suffers from exponentially decreasing acceptance rates as the dimension of the parameters grows.

In this project, we use the **Moving Average MA(2) model** as a testbed. Because the exact likelihood of MA(2) is computable via Cholesky decomposition, it provides a rigorous "Gold Standard" (via MCMC). We then benchmark two approximate methods against this standard:
1. **Standard ABC-Rejection:** Evaluates proposals jointly using a 2D Euclidean distance.
2. **ABC-Gibbs:** Updates parameters conditionally using 1D local distances, transforming a restrictive multi-dimensional acceptance sphere into a highly efficient 1D interval.

### ✨ Key Highlights
* **Mathematical Ground Truth:** Implementation of an exact Random Walk Metropolis-Hastings (RWMH) algorithm using exact likelihood calculation.
* **Component-wise Innovation:** Demonstration of how local summary statistics ($\hat{\rho}_1$ and $\hat{\rho}_2$) dramatically improve the acceptance rate.
* **Empirical Benchmarks:** Comprehensive comparison of CPU time vs. Inferential Accuracy, proving the superiority of ABC-Gibbs.

---

## 📂 Repository Structure
```text
📦 bayesian-abc-project
 ┣ 📂 notebooks                 # Step-by-step Jupyter Notebooks
 ┃ ┣ 📜 01_sanity_check.ipynb   # Verifying model properties & prior sampling
 ┃ ┣ 📜 02_gold_standard.ipynb  # Running the exact RWMH MCMC
 ┃ ┣ 📜 03_abc_methods.ipynb    # Implementing standard ABC & ABC-Gibbs
 ┃ ┗ 📜 04_comparison.ipynb     # Final benchmarks and tradeoff plots
 ┣ 📂 samplers                  # MCMC and ABC algorithms core logic
 ┃ ┣ 📜 base.py                 # SamplerResult dataclass & utilities
 ┃ ┣ 📜 rwmh.py                 # Exact Metropolis-Hastings implementation
 ┃ ┣ 📜 abc_rejection.py        # Standard ABC algorithm
 ┃ ┗ 📜 abc_gibbs.py            # Component-wise ABC-Gibbs algorithm
 ┣ 📂 report                    # LaTeX source files for Report & Slides
 ┣ 📜 ma_model.py               # MA(2) simulator, true likelihood, and prior definitions
 ┣ 📜 requirements.txt          # Python dependencies
 ┗ 📜 README.md