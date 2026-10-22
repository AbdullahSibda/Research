"""Unimodal native mixed-variable ground-truth generator.

Selects the unimodal regime of NativeMixedGenerator, retaining seven continuous
nodes and eleven genuinely categorical nodes. Continuous children follow
Gaussian conditional regressions; discrete children follow categorical softmax
models with appropriate parent-type effects. The shared generator core provides
seeded DAG construction, sampling, true-parameter export, and exact likelihoods.
"""


from _mixed_generator_core import NativeMixedGenerator

class NativeMixedUnimodalGenerator(NativeMixedGenerator):
    MODE = "unimodal"

__all__ = ["NativeMixedUnimodalGenerator"]
