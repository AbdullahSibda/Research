"""Skewed native mixed-variable ground-truth generator.

Selects the skew regime of NativeMixedGenerator for an 18-node DAG with seven
continuous and eleven categorical variables. Continuous children use skew-normal
conditional distributions while discrete children retain native categorical
softmax models. The shared core handles seeded graph generation, sampling,
probability evaluation, and export of ground-truth parameters.
"""


from _mixed_generator_core import NativeMixedGenerator

class NativeMixedSkewedGenerator(NativeMixedGenerator):
    MODE = "skew"

__all__ = ["NativeMixedSkewedGenerator"]
