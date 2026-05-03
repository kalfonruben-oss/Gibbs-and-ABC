"""
samplers/
---------
Les trois algorithmes à comparer :
  - rwmh        : Random Walk Metropolis-Hastings (gold standard)
  - abc_reject  : ABC-rejet standard
  - abc_gibbs   : ABC-Gibbs (Clarté et al.)
"""

from .base import SamplerResult
from .rwmh import run as run_rwmh, effective_sample_size
from .abc_reject import ABCRejectSampler
from .abc_gibbs import ABCGibbsSampler

__all__ = [
    "SamplerResult",
    "run_rwmh",
    "effective_sample_size",
    "ABCRejectSampler",
    "ABCGibbsSampler",
]
