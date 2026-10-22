"""dd.py — Discrete child with only discrete parents: Dirichlet-multinomial local evidence and associated count/parent-configuration handling.

All executable definitions are retained from the research implementation;
only explanatory comments and internal docstrings are consolidated here.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import gammaln


class DDScorer:
    family_name = "discrete_cpt"

    def __init__(self, data: pd.DataFrame, cardinalities: Mapping[str, int], *, equivalent_sample_size: float = 2.0) -> None:
        self.data = data
        self.cardinality = {str(k): int(v) for k, v in cardinalities.items()}
        self.equivalent_sample_size = float(equivalent_sample_size)
        if self.equivalent_sample_size <= 0:
            raise ValueError("equivalent_sample_size must be positive.")

    def score(self, child: str, d_parents: Sequence[str]) -> Tuple[float, int, int, int]:
        d_parents = tuple(sorted(map(str, d_parents)))
        r = self.cardinality[child]
        q = self._configuration_count(d_parents)
        alpha_j = self.equivalent_sample_size / q
        alpha_jk = self.equivalent_sample_size / (q * r)
        total, observed = 0.0, 0
        for _, group in self._iter_groups(d_parents):
            observed += 1
            counts = np.bincount(group[child].to_numpy(np.int64), minlength=r)
            n_j = int(counts.sum())
            total += gammaln(alpha_j) - gammaln(alpha_j + n_j)
            total += float(np.sum(gammaln(alpha_jk + counts) - gammaln(alpha_jk)))
        parameter_count = q * (r - 1)
        return float(total), int(observed), int(parameter_count), int(q)

    def _configuration_count(self, d_parents: Sequence[str]) -> int:
        return int(np.prod([self.cardinality[p] for p in d_parents])) if d_parents else 1

    def _iter_groups(self, d_parents: Sequence[str]):
        if not d_parents:
            yield None, self.data
            return
        key = d_parents[0] if len(d_parents) == 1 else list(d_parents)
        yield from self.data.groupby(key, sort=False, observed=True)
