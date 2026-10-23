# Bayesian Network Structure Learning for Continuous and Mixed Data

Research code accompanying an Honours study comparing **Discrete**, **Continuous**, and **Hybrid** Bayesian-network (BN) structure learning. The repository contains synthetic benchmark experiments (RQ1 and the native-mixed extension, RQ1.5) and an application to the English Longitudinal Study of Ageing (ELSA; RQ2).

The repository provides **source code, not restricted participant data or precomputed full research results**. The original research environment's exact package versions have not been recovered; migrating and smoke-testing code does not establish full numerical reproduction of every reported result.

## Repository structure

```text
.
├── experiments/
│   ├── rq1/
│   │   ├── continuous_benchmark.py
│   │   ├── mixed_benchmark.py
│   │   ├── continuous_plots.py
│   │   └── mixed_plots.py
│   └── rq2/
│       ├── missingness_audit.py
│       ├── generate_ensembles.py
│       ├── m_convergence.py
│       ├── analyse_graph_ensembles.py
│       ├── heldout_inference.py
│       ├── advantage_plots.py
│       ├── dementia_plots.py
│       └── consensus_plots.py
├── src/
│   ├── generators/          # Synthetic ground-truth data generators
│   ├── hybrid_bn/           # Discrete, continuous and consolidated Hybrid V3 learners
│   │   └── HybridV3_CDv3Candidate/  # Modular CC/CD/DC/DD hybrid scoring engine
│   └── imputation/          # 13 variable-specific stochastic ELSA imputers
├── requirements.txt
├── README.md
├── LICENSE
└── CITATION.cff
```

The output directories shown below are **created by the scripts** and are not required to exist in a fresh checkout. The raw ELSA `.dta` file, in contrast, must be obtained independently under the applicable data-access terms.

## Installation

A Windows/Python 3.13 setup has been used for migration smoke tests. Package versions are currently unpinned; this is a **dependency list**, not a validated environment lock file. PyMC and PyTensor are included because the Hybrid implementation has PyMC/PyTensor-backed scoring components.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

For an auditable run, capture `python --version` and `python -m pip freeze > environment_freeze.txt` in your local run records. Do not assume modern dependency versions reproduce the historic full benchmark bit-for-bit.

## RQ1: Synthetic continuous-ground-truth experiments

`experiments/rq1/continuous_benchmark.py` compares all three learners on **unimodal, skew, multimodal, and skew-multimodal** continuous generators. The full experimental design uses seeds **0–9** and sample sizes **100, 250, 500, 1,000, 2,000, 3,000, 4,000, 5,000**.

The synthetic experiments evaluate graph recovery (including SHD) and probabilistic fit using a common-resolution KL comparison. The research configuration uses a maximum of **three parents**, a structure-search iteration cap of **300**, and Hybrid ADD2 disabled. Reduced test runs are useful for checking installation, but they are **not** the full reported experiment.

```powershell
python experiments/rq1/continuous_benchmark.py --help
```

The final continuous benchmark figures can be reproduced from the complete output table `master_model_results.csv`:

```powershell
python experiments/rq1/continuous_plots.py `
  --results-dir "C:\path\to\full\continuous\results" `
  --output-dir "results\rq1\figures\continuous"
```

## RQ1.5: Native mixed-variable benchmark

`experiments/rq1/mixed_benchmark.py` evaluates synthetic ground-truth networks containing both discrete and continuous variables, across the same four generator families, seeds **0–9**, and sample sizes **100, 500, 2,000, 5,000**. The frozen configuration again uses `max_parents=3`, `max_iter=300`, and Hybrid ADD2 disabled.

```powershell
python experiments/rq1/mixed_benchmark.py --preflight-only
python experiments/rq1/mixed_benchmark.py --help
```

The final mixed figures use `mixed_model_results.csv` and `mixed_quadrant_diagnostics.csv`, including a CC/CD/DC/DD directed-edge quadrant F1 plot:

```powershell
python experiments/rq1/mixed_plots.py `
  --results-dir "C:\path\to\full\mixed\results" `
  --output-dir "results\rq1\figures\mixed" `
  --quadrant-n 5000
```

## RQ2: ELSA missingness, graph ensembles and held-out inference

### Data access and local input

ELSA's harmonised Stata dataset is **not distributed in this repository**. Obtain it through an authorised source and place it locally at:

```text
Datasets/gh_elsa_h.dta
```

Alternatively, use a local Windows junction to an existing `Datasets` directory, or use `missingness_audit.py --dta` for the audit. Phase 1 expects the `Datasets/gh_elsa_h.dta` location. Do not commit the source data, individual records, completion tables, or participant-level audit CSVs.

The pipeline studies ELSA **waves 2, 4, 6 and 8**, using **17 substantive predictors**, same-wave Dementia, and process indicators including Proxy, SelfCompletion and Nurse. This is a **contemporaneous** analysis: it does not claim to predict Wave 10 Dementia from early-wave covariates. The missingness audit also explores Wave 10 outcomes as a separate diagnostic, but that does not make the principal RQ2 evaluation longitudinal forecasting.

### Run in order

Run the commands below **from the repository root**. The full default pipeline is computationally expensive; review `--help` first, and retain the frozen experimental settings for a research reproduction run.

**1. Generate the canonical participant-completeness file.** This audit preserves Stata special missing codes and creates the frozen input needed for cohort selection.

```powershell
python experiments/rq2/missingness_audit.py --help
python experiments/rq2/missingness_audit.py
```

Its output includes:

```text
DataProcessing/missingness_final_audit/12i_participant_wave_completeness_contemporaneous.csv
```

This CSV contains participant-level information and **must remain local**. The audit also produces CASP19, recall, grip-timing and dementia/missingness diagnostics. If a raw combined Wave 8/9 nurse file is unavailable, the audit cannot establish actual grip-measurement timing from the harmonised field alone.

**2. Preflight and generate leakage-safe completion/graph ensembles.** The original incomplete cohort is split into five Dementia-stratified folds **before** imputation. Each of 13 variable-specific stochastic imputers is fitted on training data only and used to make separate training/test draws. The default design generates **25 completions per fold**, with graphs learned by Discrete, Continuous and Hybrid V3 learners.

```powershell
python experiments/rq2/generate_ensembles.py --preflight-only
python experiments/rq2/generate_ensembles.py --help
python experiments/rq2/generate_ensembles.py
```

Core output:

```text
Results/MasterPipeline/CV_Ensembles/
```

For a limited smoke test instead of a full experiment:

```powershell
python experiments/rq2/generate_ensembles.py --waves 2 --m-max 1 --fold-only 1 --models discrete --workers 1
```

**3. Examine M-convergence and learned structures.** These scripts consume saved Phase 1 DAGs; they do **not** learn new graphs. Run them after completing the requested ensemble scope.

```powershell
python experiments/rq2/m_convergence.py --help
python experiments/rq2/m_convergence.py
python experiments/rq2/analyse_graph_ensembles.py
```

M-convergence examines checkpoints **M = 10, 15, 20, 25**, so a one-completion smoke test does not suffice. The graph-ensemble analysis writes structural summaries, including `ScientificResults/dementia_neighbourhood.csv`, used by the report's structural heatmap.

**4. Evaluate fixed-DAG held-out inference.** Phase 2 reloads the saved Phase 1 completions and DAGs, fits local conditional distributions to TRAIN, evaluates observed held-out outcomes, then aggregates ensemble predictive distributions. The eight pseudo-targets are **Recall, Grip, CASP19, Loneliness, Dementia, ManagingMeds, LowerMobility, CESD**. Out-of-fold predictions are stitched before final metrics, with paired bootstrap comparisons.

```powershell
python experiments/rq2/heldout_inference.py --help
python experiments/rq2/heldout_inference.py --preflight-only
python experiments/rq2/heldout_inference.py
```

Core output:

```text
Results/MasterPipeline/CV_HeldoutInference/
```

Hybrid Loneliness inference uses **256 quadrature nodes per bin (768 total)** with a chunk size of **32**, replacing the historical separate Loneliness correction script. Other targets retain the default quadrature settings. The corrected pathway is incorporated into inference with cache/provenance safeguards; its full numerical equivalence to the old corrective rerun has **not yet been established**.

A limited Phase 1 → Phase 2 handoff test can use:

```powershell
python experiments/rq2/heldout_inference.py --waves 2 --fold-only 1 --completion-only 1 --m-max 1 --models discrete --targets Dementia --skip-bootstrap --skip-plots
```

This creates a **partial** set of held-out predictions. It does not create valid full five-fold out-of-fold performance metrics or final report figures.

### Generate only the six RQ2 report figures

Run these **only after the required complete results exist**, from the repository root:

```powershell
# Figures 6–9: paired-advantage LogLoss, F1, NLPD and RMSE forests
python experiments/rq2/advantage_plots.py

# Figure 10: Dementia F1 vs decision threshold
python experiments/rq2/dementia_plots.py

# Figure 11: Dementia-neighbourhood stability heatmap
python experiments/rq2/consensus_plots.py --project-root .
```

The paired-advantage plotter reads `Results/MasterPipeline/CV_HeldoutInference/metrics/model_comparisons.csv` directly, and the Dementia plotter reads the Phase 2 Dementia threshold curves. Both write to the Phase 2 `plots/report/` directory by default. The structural heatmap reads `ScientificResults/dementia_neighbourhood.csv` and writes to `ScientificPlots/`.

The six figures retained from the research report are:

| Figure | Content |
| --- | --- |
| 6 | Hybrid paired advantage: LogLoss |
| 7 | Hybrid paired advantage: F1 (Dementia excluded) |
| 8 | Hybrid paired advantage: NLPD |
| 9 | Hybrid paired advantage: RMSE |
| 10 | Dementia F1 versus decision threshold |
| 11 | Hybrid Dementia-neighbourhood stability heatmap |

Extra exploratory network diagrams, temporal adjacency trajectories, split-wave forests and additional heatmaps have been omitted from the final plotting scripts. The forest plotter consistently interprets **positive advantage as Hybrid performing better**. The source CSV must contain the corresponding paired bootstrap confidence intervals; plotting does not rerun the bootstrap.

## Verification status

- **RQ1/RQ1.5:** Migrated single-cell smoke tests checked synthetic structure-learning interfaces and reported structural metrics. Full-precision KL and full-grid numerical equivalence have not been established.
- **RQ2 Phase 1:** Preflight passed with the local ELSA inputs and all 13 imputers/three learner modules loaded; a Wave 2, Fold 1, M 1, Discrete smoke run completed successfully.
- **RQ2 Phase 2:** The same single-cell Discrete Dementia test produced **1,833 held-out predictions**. The reconstructed representation matched, and the fixed-DAG refit score difference was approximately **1.455 × 10⁻¹¹**. Final aggregate metrics correctly remained unavailable because other folds were not run.
- **Remaining checks:** Full five-fold reproduction, Hybrid inference, corrected Loneliness numerical parity and final plots from the original completed results remain unverified in the migrated environment.

## Generated outputs and privacy

The following locations are generated locally and should be gitignored: `Datasets/`, `DataProcessing/`, `Results/` (or `results/`), `ScientificResults/`, `ScientificPlots/`, Python caches and virtual environments. **Only generated directories are recreated automatically; raw ELSA data are not downloaded or generated by the code.** The missingness audit must run before Phase 1 unless its canonical cohort file has already been produced locally. If a path is ignored after files have already been committed, `.gitignore` alone does not erase those files from Git history.

## License and citation

Consult the repository's `LICENSE` and `CITATION.cff` for reuse and citation information. These apply to the original research code, **not** to redistribution of the restricted ELSA dataset.
