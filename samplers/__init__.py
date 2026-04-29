"""
samplers/
---------
Les trois algorithmes à comparer :
  - rwmh        : Random Walk Metropolis-Hastings (gold standard)
  - abc_reject  : ABC-rejet standard
  - abc_gibbs   : ABC-Gibbs (Clarté et al.)
"""

from samplers.base import SamplerResult