# Bayesian Network Structure Learning for Mixed Data

Research code for comparing **discrete**, **continuous**, and **hybrid** Bayesian-network structure learning, with synthetic benchmarking and a planned application to longitudinal ageing data.

> **Repository status (October 2026):** The RQ1/RQ1.5 benchmark and plotting scripts have been migrated and smoke-tested. RQ2 integration and its documentation are forthcoming. This README documents the available RQ1 workflow; it does not imply that the entire research pipeline has already been migrated.

## Research overview

The project investigates how different Bayesian-network representations behave when discovering dependency structures in data containing continuous and discrete variables.

- **Discrete baseline:** A fully discrete approach, including discretisation when the ground truth is continuous.
- **Continuous baseline:** A continuous-variable structure-learning approach.
- **Hybrid approach:** A mixed-variable Bayesian network with conditional scoring for continuous-to-continuous (CC), continuous-to-discrete (CD), discrete-to-continuous (DC), and discrete-to-discrete (DD) node families.

The benchmarks compare graph recovery and probabilistic fit, including **structural Hamming distance (SHD)** and **KL divergence**. The common-resolution KL evaluation places the three representations on a shared comparison basis. The reported structural metrics measure recovery of the known ground-truth graph in synthetic experiments.

## Repository layout

```text
.
├── experiments/
│   └── rq1/
│       ├── continuous_benchmark.py
│       ├── mixed_benchmark.py
│       ├── continuous_plots.py
│       └── mixed_plots.py
├── src/
│   ├── hybrid_bn/          # Structure-learning and scoring implementations
│   └── generators/         # Synthetic ground-truth generators
├── requirements.txt
└── README.md
```

The diagram shows the intended RQ1 source layout; generated result folders and any additional repository files are omitted. The benchmark scripts contain their evaluation routines as readable, namespaced Python code and use model/generator implementations under `src/`.

## Requirements and setup

- **Python 3.13** is the version used for the post-migration smoke tests on Windows. Exact historical package versions for the original research runs have not yet been established.
- The RQ1 `requirements.txt` includes: `numpy`, `pandas`, `scipy`, `scikit-learn`, `networkx`, `matplotlib`, and `seaborn`.

From the repository root, create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell activation is restricted, invoke `.\.venv\Scripts\python.exe` directly. The scripts also accept `--project-root` to specify the repository root explicitly.

## Research Question 1: Continuous ground-truth benchmark

**Runner:** `experiments/rq1/continuous_benchmark.py`

Four synthetic continuous-generator families are studied: **unimodal**, **skew**, **multimodal**, and **skew-multimodal** (`skew_multi`). The complete benchmark uses seeds 0–9 and sample sizes **100, 250, 500, 1,000, 2,000, 3,000, 4,000, and 5,000**. It evaluates the continuous, discrete, and hybrid learners against the known ground-truth structures.

The frozen comparison uses a maximum of **three parents**, structure-search caps of **300 iterations** for the baseline and hybrid models, and **Hybrid ADD2 disabled**. Do not change these settings if attempting to reproduce the research configuration.

Inspect the supported options before running:

```powershell
python experiments/rq1/continuous_benchmark.py --help
```

A complete benchmark can be computationally expensive. For a restricted run, specify `--generators`, `--seeds`, and `--sample-sizes`; preserve the frozen structure-search configuration. Monte Carlo evaluation options are exposed separately for diagnostic runs, but reducing them changes KL-estimate precision.

## Research Question 1.5: Native mixed-variable benchmark

**Runner:** `experiments/rq1/mixed_benchmark.py`

This benchmark uses synthetic data with **discrete and continuous nodes in the ground truth**. The design covers the same four generator families, seeds 0–9, and sample sizes **100, 500, 2,000, and 5,000**. It compares the same three learning paradigms using graph-recovery and probabilistic metrics.

The frozen structure-learning contract requires **`max_parents=3`, `max_iter=300`, and Hybrid ADD2 disabled**. The runner rejects incompatible settings (for example, `--max-iter 10`).

Check dependencies without launching a full experiment:

```powershell
python experiments/rq1/mixed_benchmark.py --preflight-only
```

View all controls:

```powershell
python experiments/rq1/mixed_benchmark.py --help
```

Use `--no-legacy-reuse` when you want to rule out reusing existing legacy results. Because this benchmark can be expensive, inspect and set the output directory before starting a complete run.

## Reproduce the five final RQ1 figures

Plotting reads **completed benchmark CSVs**; it does not retrain models. The one-cell smoke-test data are insufficient for reproducing full learning curves.

| Script | Figures produced | Required input files |
| --- | --- | --- |
| `continuous_plots.py` | Common-resolution KL (4 panels); SHD (4 panels) | `master_model_results.csv` |
| `mixed_plots.py` | Common-support KL (4 panels); SHD (4 panels); directed-edge quadrant F1 | `mixed_model_results.csv`, `mixed_quadrant_diagnostics.csv` |

Examples from the repository root (replace the result directories with the paths containing your **full** experiment outputs):

```powershell
python experiments/rq1/continuous_plots.py `
  --results-dir "C:\path\to\full\continuous\results" `
  --output-dir "results\rq1\figures\continuous"

python experiments/rq1/mixed_plots.py `
  --results-dir "C:\path\to\full\mixed\results" `
  --output-dir "results\rq1\figures\mixed" `
  --quadrant-n 5000
```

The scripts write a **PNG and vector PDF** of each figure. The quadrant F1 figure summarises directed-edge recovery across CC, CD, DC, and DD categories at the selected sample size.

## Migration validation and reproducibility limits

The migrated continuous and native-mixed runners both passed an end-to-end **single-cell smoke test** with `unimodal`, seed `0`, and sample size `100`. The matching original and migrated results agreed exactly on the **reported structural metrics** for all three learner types in both benchmarks, including SHD and F1. Hybrid structural quadrant diagnostics also matched for the continuous benchmark test cell.

The smoke tests deliberately used lower Monte Carlo sample counts for probabilistic evaluations. **Full-precision KL numerical parity was not established** by these tests, and the smoke-test KL values should **not** replace the original full experimental results. Information-retention diagnostics are not among the five report figures documented here.

For a reproducible comparison, record the run arguments, seeds, full dependency versions, result manifests, and input hashes. Do not compare KL estimates across runs with different evaluation settings as though they were identically configured.

## Research Question 2

RQ2 applies the methods to longitudinal ageing data, including preprocessing, missing-data handling, Bayesian-network ensembles, and downstream analyses. **The RQ2 code migration is pending**; commands and the final dependency list will be added after that workflow has been checked in the repository.

## Notes on data and outputs

Synthetic benchmark output files, logs, cached model fits, and figures are generated artifacts. Keep large or restricted datasets out of Git unless you have confirmed that redistribution is permitted. The dataset and output retention policy should be reflected in `.gitignore` before publishing additional research artifacts.

## License and citation

A license and formal citation details have not yet been specified in the repository materials reviewed for this README. Add them once the intended reuse terms and final research-report citation are confirmed.
