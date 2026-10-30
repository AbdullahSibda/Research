"""
Expected source files:
  src/hybrid_bn/SLhybrid.py and src/hybrid_bn/HybridV3_CDv3Candidate/
  src/hybrid_bn/SLdiscrete.py and src/hybrid_bn/SLcontinuous.py (or src/ equivalents)

This is a *new variable-size native-mixed generator* for scaling, not the fixed
18-node RQ1.5 ground-truth generator. All three models see the same generated
observations, using model-appropriate representations. No accuracy claims.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'src' / 'hybrid_bn' / 'SLhybrid.py').exists():
            return parent
    raise RuntimeError('Cannot locate repository root: expected src/hybrid_bn/SLhybrid.py. Pass --project-root.')


def module_file(root: Path, name: str) -> Path:
    for candidate in (root / 'src' / 'hybrid_bn' / (name + '.py'), root / 'src' / (name + '.py')):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Cannot find {name}.py in src/hybrid_bn/ or src/')


def load_models(root: Path, requested):
    bn_dir = root / 'src' / 'hybrid_bn'
    src_dir = root / 'src'
    for folder in (bn_dir, src_dir):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    models = {}
    for choice, name in (('hybrid', 'SLhybrid'), ('discrete', 'SLdiscrete'), ('continuous', 'SLcontinuous')):
        if choice in requested:
            file = module_file(root, name)
            spec = importlib.util.spec_from_file_location(name, file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            models[choice] = mod
    return models


def generate(p: int, n: int, seed: int, max_parents: int):
    """A variable-node native-mixed DAG; both C->D and D->C possible.

    Nodes alternate C and D (three-state). Edge probability scales as 2.5/p;
    in-degree is capped. CPDs use linear mean effects and latent categorical
    thresholds (not post-hoc discretisation of a continuous ground-truth node).
    """
    rng = np.random.default_rng(seed)
    names = [f'X{i:02d}' for i in range(p)]
    continuous = names[::2]
    discrete = names[1::2]
    discrete_set = set(discrete)
    dag = nx.DiGraph()
    dag.add_nodes_from(names)
    values = {}
    probability = min(1.0, 2.5 / max(1, p - 1))
    for j, child in enumerate(names):
        chosen = [node for node in names[:j] if rng.random() < probability]
        if len(chosen) > max_parents:
            chosen = list(rng.choice(chosen, max_parents, replace=False))
        dag.add_edges_from((node, child) for node in chosen)
        if chosen:
            x = np.column_stack([values[node] for node in chosen]).astype(float)
            x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 0.3)
            weights = rng.uniform(-0.9, 0.9, len(chosen))
            eta = x @ weights
        else:
            eta = np.zeros(n)
        if child in discrete_set:
            # Native ordered categorical conditional, represented as 0/1/2.
            latent = eta + rng.normal(0, 1, n)
            values[child] = np.digitize(latent, [-0.6, 0.6]).astype(int)
        else:
            values[child] = eta + rng.normal(0, 1, n)
    return pd.DataFrame(values)[names], dag, continuous, discrete


def training_representation(raw, continuous, discrete, seed, model):
    result = raw.copy()
    cards = {node: 3 for node in discrete}
    if model == 'continuous':
        # Same dequantisation policy as native-mixed RQ1.5 comparisons.
        for i, node in enumerate(discrete):
            rng = np.random.default_rng(seed * 10009 + i + 71)
            result[node] = result[node].astype(float) + rng.uniform(-0.5, 0.5, len(result))
    elif model == 'discrete':
        for node in continuous:
            # Deterministic rank-quantile categories, using training rows only.
            result[node] = pd.qcut(result[node].rank(method='first'), q=3, labels=False).astype(int)
        cards.update({node: 3 for node in continuous})
    return result, cards


def run_learner(name, model, raw, continuous, discrete, opt, seed, cc_file):
    data, cards = training_representation(raw, continuous, discrete, seed, name)
    nodes = list(raw.columns)
    start = time.perf_counter()
    if name == 'continuous':
        scorer = model.ContinuousScoringEngine(data)
        search = model.TabuGreedyHillClimbing(nodes, scorer, tabu_length=opt.tabu_length,
            max_iter=opt.max_iter, max_parents=opt.max_parents, improvement_tolerance=1e-10)
    elif name == 'discrete':
        scorer = model.DiscreteScoringEngine(data, cards)
        search = model.TabuGreedyHillClimbing(nodes, scorer, tabu_length=opt.tabu_length,
            max_iter=opt.max_iter, max_parents=opt.max_parents, improvement_tolerance=1e-10)
    else:
        scorer_type = getattr(model, 'HybridV3ScoringEngine', None) or getattr(model, 'HybridScoringEngine')
        scorer = scorer_type(data, discrete_nodes=discrete, continuous_nodes=continuous,
                            cardinalities={node: cards[node] for node in discrete},
                            cc_scorer_file=cc_file)
        # Use standard single-edge tabu (no ADD2), as in the frozen benchmark.
        search = model.TabuGreedyHillClimbing(nodes, scorer, tabu_length=opt.tabu_length,
            max_iter=opt.max_iter, max_parents=opt.max_parents, improvement_tolerance=1e-10)
    edges = search.search()
    runtime = time.perf_counter() - start
    return dict(runtime_seconds=runtime, learned_edges=len(edges),
                search_iterations=max(0, len(getattr(search, 'score_history_', [])) - 1),
                family_cache_size=len(getattr(scorer, 'cache', {})))


def summary(output: Path):
    records = []
    for path in sorted((output / 'runs').glob('*.json')):
        record = json.loads(path.read_text(encoding='utf8'))
        if record.get('status') == 'complete':
            records.append(record)
    if records:
        frame = pd.DataFrame(records).sort_values(['nodes', 'seed', 'model'])
        frame.to_csv(output / 'scalability_runs.csv', index=False)
        (frame.groupby(['nodes', 'model'], as_index=False)
         .agg(replicates=('seed', 'count'), median_seconds=('runtime_seconds', 'median'),
              mean_seconds=('runtime_seconds', 'mean'))
         .to_csv(output / 'scalability_summary.csv', index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=None)
    parser.add_argument('--nodes', nargs='+', type=int, default=[10, 15, 20])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--max-iter', type=int, default=150)
    parser.add_argument('--max-parents', type=int, default=3)
    parser.add_argument('--tabu-length', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1, help='Reserved: learning runs serially for comparable timing.')
    parser.add_argument('--models', nargs='+', choices=['continuous', 'discrete', 'hybrid'],
                        default=['continuous', 'discrete', 'hybrid'])
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    opt = parser.parse_args()
    root = (opt.project_root or repository_root()).resolve()
    if not (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').exists():
        parser.error('Expected src/hybrid_bn/SLhybrid.py under --project-root.')
    if min(opt.nodes) < 4 or opt.samples < 10 or opt.max_parents < 1:
        parser.error('Require nodes >= 4, samples >= 10, max_parents >= 1.')
    if opt.workers != 1:
        print('NOTE: --workers is not used by this serial, single-process learner.', flush=True)
    files = [module_file(root, 'SL' + name) for name in opt.models]
    print('Repository:', root, '\nLearners:', ', '.join(str(p) for p in files),
          '\nDesign: variable-size native-mixed (not the fixed RQ1.5 truth generator)',
          '\nGrid:', opt.nodes, 'x', opt.seeds, 'at n=', opt.samples, flush=True)
    if 'hybrid' in opt.models:
        bn = root / 'src' / 'hybrid_bn'
        cc_candidate = bn / 'SLcontinuousV3Final_CorrectedCandidateV2.py'
        cc_production = bn / 'SLcontinuousV3Final.py'
        if cc_candidate.exists() and not cc_production.exists() and not os.environ.get('V3_PRODUCTION_SCORER'):
            print('WARNING: legacy CC candidate may require SLcontinuousV3Final.py.', flush=True)
    if opt.dry_run:
        # Validate imports and public engine/search names.
        models = load_models(root, opt.models)
        for name, module in models.items():
            engine = {'hybrid': ('HybridV3ScoringEngine', 'HybridScoringEngine'),
                      'discrete': ('DiscreteScoringEngine',),
                      'continuous': ('ContinuousScoringEngine',)}[name]
            if not any(hasattr(module, candidate) for candidate in engine):
                raise RuntimeError(f'{name}: scoring API not available on {module.__file__}')
            if not hasattr(module, 'TabuGreedyHillClimbing'):
                raise RuntimeError(f'{name}: TabuGreedyHillClimbing not found')
        print('PREFLIGHT PASSED (imports and class names). No learning run performed.', flush=True)
        return
    out = (opt.output_dir or root / 'outputs' / 'scalability').resolve()
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    models = load_models(root, opt.models)
    cc_file = None
    for filename in ('SLcontinuousV3Final_CorrectedCandidateV2.py', 'SLcontinuousV3Final.py'):
        path = root / 'src' / 'hybrid_bn' / filename
        if path.exists():
            cc_file = path
            break
    # Under consolidated deployments the CC implementation may resolve by default.
    for size in sorted(set(opt.nodes)):
        for seed in opt.seeds:
            raw, truth, continuous, discrete = generate(size, opt.samples, seed, opt.max_parents)
            for name in opt.models:
                target = out / 'runs' / f'p{size:03d}_n{opt.samples}_seed{seed:03d}_{name}.json'
                if target.exists() and not opt.overwrite:
                    old = json.loads(target.read_text(encoding='utf8'))
                    if old.get('status') == 'complete':
                        print('SKIP completed', target.name, flush=True)
                        continue
                record = dict(status='complete', model=name, nodes=size, seed=seed,
                    samples=opt.samples, generator='variable_native_mixed_unimodal',
                    ground_truth_edges=truth.number_of_edges(), max_iter=opt.max_iter,
                    max_parents=opt.max_parents, tabu_length=opt.tabu_length,
                    workers=1, python=platform.python_version(), platform=platform.platform())
                print('RUN', name, 'p=', size, 'seed=', seed, flush=True)
                try:
                    record.update(run_learner(name, models[name], raw, continuous, discrete,
                                              opt, seed, cc_file))
                    print('DONE', name, f"{record['runtime_seconds']:.3f}s", flush=True)
                except Exception as error:
                    record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
                    print('FAILED', name, repr(error), flush=True)
                temp = target.with_suffix('.tmp')
                temp.write_text(json.dumps(record, indent=2), encoding='utf8')
                os.replace(temp, target)
                summary(out)
    print('Outputs:', out, flush=True)

if __name__ == '__main__':
    main()
