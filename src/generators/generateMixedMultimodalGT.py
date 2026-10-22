"""Multimodal native mixed-variable ground-truth generator.

Selects the multimodal regime of NativeMixedGenerator for seven continuous and
eleven native categorical nodes. Continuous roots can follow Gaussian mixtures;
conditional continuous nodes include threshold-gated shifts. Discrete children
may depend nonlinearly on continuous parents through quadratic and threshold
features. Seeded graph generation, exact likelihoods, and sampling are supplied
by the common generator core.
"""


from _mixed_generator_core import NativeMixedGenerator

class NativeMixedMultimodalGenerator(NativeMixedGenerator):
    MODE = "multimodal"

__all__ = ["NativeMixedMultimodalGenerator"]
