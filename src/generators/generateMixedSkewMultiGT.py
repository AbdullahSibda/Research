"""Skewed multimodal native mixed-variable ground-truth generator.

Selects the skew_multi regime of NativeMixedGenerator on an 18-node graph with
seven continuous and eleven categorical variables. Continuous roots may follow
skew-normal mixtures and conditional continuous children combine skewed noise
with threshold gates. Discrete children use native categorical softmax models,
including nonlinear continuous-parent effects. The core provides reproducible
sampling, ground-truth parameter export, and probability evaluation.
"""


from _mixed_generator_core import NativeMixedGenerator

class NativeMixedSkewedMultimodalGenerator(NativeMixedGenerator):
    MODE = "skew_multi"

__all__ = ["NativeMixedSkewedMultimodalGenerator"]
