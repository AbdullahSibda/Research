"""Continuous-child / continuous-parent quadrant adapter.

Provides CCScorer to the modular Hybrid dispatcher. The adapter loads the
single-file consolidated V3 master by explicit path or by searching beside
this package; HYBRID_V3_CC_SCORER can override discovery. Its interface exposes
local scores, joint log densities, cached values and results without changing
the underlying corrected continuous-family numerical implementation.
"""





from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def discover_cc_scorer(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    env = os.environ.get("HYBRID_V3_CC_SCORER")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "SLhybrid.py",
        here.parent / "SLhybrid.py",
        Path.cwd() / "Models" / "SLhybrid.py",
        Path.cwd() / "SLhybrid.py",
    ]
    for p in candidates:
        p = p.resolve()
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate SLhybrid.py. "
        "Put SLhybrid.py in src/hybrid_bn/,, set HYBRID_V3_CC_SCORER, "
        "or pass cc_scorer_file=... ."
    )


class CCScorer:
    family_name = "continuous_v3_corrected"

    def __init__(self, raw_continuous_data: pd.DataFrame, *, scorer_file: str | Path | None = None) -> None:
        if raw_continuous_data.empty:
            raise ValueError("CCScorer requires at least one continuous node.")
        self.scorer_file = discover_cc_scorer(scorer_file)
        self.module = _load_module(
            self.scorer_file,
            f"_hybrid_v3_cc_{os.getpid()}_{id(self)}",
        )
        if not hasattr(self.module, "V3CScoringEngine"):
            raise AttributeError(f"{self.scorer_file} does not expose V3CScoringEngine.")
        self.engine = self.module.V3CScoringEngine(raw_continuous_data.copy())

    def score(self, child: str, c_parents: Sequence[str]) -> float:
        return float(self.engine.score_local(str(child), tuple(sorted(map(str, c_parents)))))

    def joint_logpdf(self, graph, evaluation_data: pd.DataFrame):
        return self.engine.joint_logpdf(graph, evaluation_data)

    @property
    def cache(self):
        return getattr(self.engine, "cache", {})

    @property
    def results(self):
        return getattr(self.engine, "results", {})
