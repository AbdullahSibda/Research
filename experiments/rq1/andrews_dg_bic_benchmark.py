"""Andrews et al. (2019) Degenerate Gaussian BIC + FGES literature baseline.

Run from repository root, in .venv-tetrad:
  python -u experiments/rq1/andrews_dg_bic_benchmark.py --preflight
  python -u experiments/rq1/andrews_dg_bic_benchmark.py --generators unimodal --seeds 0 --sample-sizes 500
  python -u experiments/rq1/andrews_dg_bic_benchmark.py --generators unimodal skew multimodal skew_multi --seeds 0 1 2 --sample-sizes 500 2000 5000

Regenerates the SAME deterministic native-mixed data and row prefixes used by
mixed_benchmark.py (generator/seed, sample_seed=100).
It does NOT retrain Hybrid. Compare to saved Hybrid results only after verifying
identical generators/seeds/n/graph design and data hashes where available.
FGES outputs a CPDAG: only skeleton metrics are computed here. No arbitrary
orientation is treated as an inferred direction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SPECS = {
    'unimodal': ('generateMixedUnimodalGT.py', 'NativeMixedUnimodalGenerator'),
    'skew': ('generateMixedSkewGT.py', 'NativeMixedSkewedGenerator'),
    'multimodal': ('generateMixedMultimodalGT.py', 'NativeMixedMultimodalGenerator'),
    'skew_multi': ('generateMixedSkewMultiGT.py', 'NativeMixedSkewedMultimodalGenerator'),
}

def repository_root():
    here = Path(__file__).resolve()
    for root in (here.parent, *here.parents, Path.cwd(), *Path.cwd().parents):
        if (root / 'src' / 'generators' / '_mixed_generator_core.py').exists():
            return root
    raise FileNotFoundError('Cannot find src/generators/_mixed_generator_core.py; run within Research repository.')

def load_generator(root: Path, key: str):
    directory = root / 'src' / 'generators'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    filename, classname = SPECS[key]
    spec = importlib.util.spec_from_file_location('dg_generator_' + key, directory / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, classname)

def data_hash(frame):
    # Exact fingerprint of typed values in consistent row/column order; for provenance.
    h = hashlib.sha256()
    h.update('|'.join(frame.columns).encode())
    h.update(np.ascontiguousarray(frame.to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()

def extract_edges(graph_str: str, node_names: set[str]):
    """Read only 'Graph Edges:' section. Preserve every endpoint mark."""
    m = re.search(r'Graph Edges:\s*(.*?)(?:\n\s*Graph Attributes:|\n\s*Graph Node Attributes:|\Z)', graph_str, re.S)
    if not m:
        raise ValueError('Cannot parse Tetrad graph edge section: ' + graph_str[:400])
    text = m.group(1)
    edges = []
    for line in text.splitlines():
        line = re.sub(r'^\s*\d+\.\s*', '', line.strip())
        if not line:
            continue
        match = re.fullmatch(r'(.+?)\s+(-->|<--|<->|o->|<-o|o-o|---)\s+(.+?)', line)
        if not match:
            raise ValueError('Unrecognised Tetrad edge notation: ' + repr(line))
        a, mark, b = (x.strip() for x in match.groups())
        if a not in node_names or b not in node_names or a == b:
            raise ValueError(f'Unexpected graph nodes/loop: {a!r} {mark} {b!r}')
        edges.append({'node_a': a, 'endpoint_mark': mark, 'node_b': b})
    pairs = [frozenset((e['node_a'], e['node_b'])) for e in edges]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Tetrad returned repeated adjacency')
    return edges

def eval_skeleton(truth_graph, estimated_edges):
    true = {frozenset((str(a), str(b))) for a, b in truth_graph.edges()}
    pred = {frozenset((e['node_a'], e['node_b'])) for e in estimated_edges}
    tp = len(true & pred)
    fp = len(pred - true)
    fn = len(true - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {'true_skeleton_edges': len(true), 'estimated_skeleton_edges': len(pred),
            'skeleton_tp': tp, 'skeleton_fp': fp, 'skeleton_fn': fn,
            'skeleton_precision': precision, 'skeleton_recall': recall,
            'skeleton_f1': f1, 'skeleton_shd': fp + fn,
            'undirected_or_partially_oriented': sum(e['endpoint_mark'] != '-->' and e['endpoint_mark'] != '<--' for e in estimated_edges)}

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project-root', type=Path)
    ap.add_argument('--output-root', type=Path)
    ap.add_argument('--generators', nargs='+', choices=list(SPECS), default=['unimodal'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--sample-sizes', nargs='+', type=int, default=[500])
    ap.add_argument('--max-degree', type=int, default=3)
    ap.add_argument('--penalty-discount', type=float, default=1.0)
    ap.add_argument('--structure-prior', type=float, default=0.0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--continue-on-error', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()
    root = args.project_root.resolve() if args.project_root else repository_root()
    out = args.output_root.resolve() if args.output_root else root / 'outputs' / 'rq1_andrews_dg_bic'
    if min(args.sample_sizes) <= 0 or args.max_degree < 1:
        ap.error('sample sizes and --max-degree must be positive')
    for key in args.generators:
        cls = load_generator(root, key)
        print(f'[GENERATOR] {key}: {cls.__name__}', flush=True)
    if args.preflight:
        print('[PREFLIGHT] Generator paths OK; actual algorithm needs py-tetrad and JVM at execution.')
        return
    from pytetrad.tools.TetradSearch import TetradSearch
    import jpype
    import importlib.metadata
    print(f'[DG-BIC] py-tetrad={importlib.metadata.version("py-tetrad")}, JPype={jpype.__version__}; root={root}', flush=True)
    print(f'[DG-BIC] FGES degree={args.max_degree}, penalty={args.penalty_discount}, prior={args.structure_prior}', flush=True)
    for key in args.generators:
        Generator = load_generator(root, key)
        for seed in args.seeds:
            max_n = max(args.sample_sizes)
            gt = Generator(num_samples=max_n, seed=seed)
            gt.generate_dag()
            raw = pd.DataFrame(gt.generate_data(), columns=gt.nodes).reset_index(drop=True)
            for n in sorted(set(args.sample_sizes)):
                target_dir = out / 'runs' / key / f'seed_{seed:02d}' / f'n_{n:04d}'
                result_file = target_dir / 'result.json'
                frame = raw.iloc[:n].copy()
                for col in gt.continuous_nodes:
                    frame[col] = pd.to_numeric(frame[col]).astype('float64')
                for col in gt.discrete_nodes:
                    vals = pd.to_numeric(frame[col])
                    if vals.isna().any() or not np.all(vals.to_numpy() == vals.to_numpy().astype(np.int64)):
                        raise ValueError(f'Invalid categorical values in {col}')
                    frame[col] = vals.astype('int64')
                digest = data_hash(frame)
                if result_file.exists() and not args.overwrite:
                    saved = json.loads(result_file.read_text(encoding='utf-8'))
                    if saved.get('data_sha256') != digest:
                        raise RuntimeError(f'Data changed for {key}/seed{seed}/n{n}; do not silently reuse')
                    print(f'[REUSE] {key} seed={seed} n={n} status={saved.get("status")}', flush=True)
                    continue
                print(f'[START] {key} seed={seed} n={n}', flush=True)
                started = time.perf_counter()
                try:
                    search = TetradSearch(frame)
                    search.use_degenerate_gaussian_score(
                        penalty_discount=args.penalty_discount,
                        structure_prior=args.structure_prior,
                    )
                    search.run_fges(max_degree=args.max_degree, parallelized=False)
                    graph_text = str(search.get_string())
                    edges = extract_edges(graph_text, set(gt.nodes))
                    metrics = eval_skeleton(gt.graph, edges)
                    record = {
                        'status': 'ok', 'method': 'Andrews 2019 DG-BIC + Tetrad FGES',
                        'generator': key, 'seed': seed, 'n': n,
                        'data_sha256': digest, 'node_count': len(gt.nodes),
                        'continuous_nodes': list(gt.continuous_nodes),
                        'discrete_nodes': list(gt.discrete_nodes),
                        'max_degree': args.max_degree, 'penalty_discount': args.penalty_discount,
                        'structure_prior': args.structure_prior,
                        'runtime_seconds': time.perf_counter()-started,
                        'python_version': platform.python_version(),
                        'py_tetrad_version': importlib.metadata.version('py-tetrad'),
                        'jpype_version': jpype.__version__,
                        'java_version': str(jpype.java.lang.System.getProperty('java.version')) if jpype.isJVMStarted() else 'unknown',
                        'graph_output_type': 'FGES CPDAG (may contain undirected edges)',
                        **metrics,
                    }
                    target_dir.mkdir(parents=True, exist_ok=True)
                    (target_dir / 'tetrad_graph.txt').write_text(graph_text, encoding='utf-8')
                    pd.DataFrame(edges, columns=['node_a','endpoint_mark','node_b']).to_csv(target_dir / 'estimated_edges.csv', index=False)
                    pd.DataFrame(list(gt.graph.edges()), columns=['parent','child']).to_csv(target_dir / 'truth_edges.csv', index=False)
                    write_json(result_file, record)
                    print(f'[OK] skeleton F1={metrics["skeleton_f1"]:.3f} SHD={metrics["skeleton_shd"]} time={record["runtime_seconds"]:.1f}s', flush=True)
                except Exception as exc:
                    record = {'status': 'error','generator': key,'seed': seed,'n': n,
                              'data_sha256': digest, 'error': str(exc), 'traceback': traceback.format_exc()}
                    write_json(result_file, record)
                    print(f'[ERROR] {key}/seed{seed}/n{n}: {exc}', flush=True)
                    if not args.continue_on_error:
                        raise
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'runs').rglob('result.json'))]
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('continuous_nodes','discrete_nodes','traceback')}
                  for r in records]).to_csv(out/'dg_bic_results.csv', index=False)
    print(f'[DONE] {len(records)} result checkpoints; summary: {out / "dg_bic_results.csv"}', flush=True)

if __name__ == '__main__':
    main()
