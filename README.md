<p align="center">
  <img src="assets/logo.png" alt="M3TRICS" width="360">
</p>

M3TRICS is a multimodal training and analysis framework for comparing missing-modality learning methods across clinical datasets. It handles dataset preprocessing, modality alignment, nested cross-validation, progressive missingness studies, fixed-dataset training, binary classification, survival modelling, and downstream statistical analysis.

This README describes the workflow from environment setup to result analysis. The project root referred to below is this directory, the one containing `scripts/m3trics.py` and the `run_M3TRICS_*.sh` scripts.

## What This Project Can Do

- Train multimodal models from per-modality CSV files.
- Align endpoint and modality tables by patient ID.
- Drop dataset-specific columns before training.
- Aggregate duplicated patient rows with mean pooling or supervised attention pooling.
- Impute missing values within each modality using configurable numeric/categorical strategies.
- Compare multiple missing-modality methods under the same nested-CV setup.
- Run either a progressive missingness study or a static-cohort experiment.
- Evaluate binary classification and discrete-time survival tasks.
- Save inner-CV, outer-test, epoch-history, split, prediction, and processed-data outputs.
- Analyze trained results with notebooks in `analysis/`.

## Repository Layout

```text
m3trics/
├── run_M3TRICS_mmImmuno.sh                # mmImmuno training launcher
├── run_M3TRICS_mmColorectal.sh               # mmColorectal training launcher
├── run_M3TRICS_mmProstate.sh               # mmProstate training launcher
├── env/                               # Conda environment definitions and install guide
├── hyperparams/                       # Hyperparameter grids per method
├── dataset/                           # Dataset preprocessing, loaders, missingness simulator, pooling, imputation
├── scripts/                           # CLI entrypoint, nested-CV training logic, training loops, shared utilities
├── models/                            # Model implementations
├── analysis/                          # Result analysis notebooks and helper code
└── results/                           # Generated processed data, model outputs, W&B logs
```

## Available Methods

Use these names in `RUN_MODELS` inside the `.sh` launchers.

| Method | Purpose |
| --- | --- |
| `ZI_MLP` | MultimodalMLP with zero imputation for missing modalities. |
| `KNN_MLP` | MultimodalMLP with KNN imputation for missing modalities. |
| `ZI_LR` | Logistic regression baseline over concatenated modality features with zero imputation for missing modalities. Classification only. |
| `KNN_LR` | Logistic regression baseline over concatenated modality features with KNN imputation for missing modalities. Classification only. |
| `ZI_RF` | Random forest classifier baseline over concatenated modality features with zero imputation. Classification only. |
| `KNN_RF` | Random forest classifier baseline over concatenated modality features with KNN imputation. Classification only. |
| `ZI_CoxNet` | Regularized Cox elastic-net baseline over concatenated modality features with zero imputation. Survival only. |
| `KNN_CoxNet` | Regularized Cox elastic-net baseline over concatenated modality features with KNN imputation. Survival only. |
| `ZI_RSF` | Random Survival Forest baseline over concatenated modality features with zero imputation. Survival only. |
| `KNN_RSF` | Random Survival Forest baseline over concatenated modality features with KNN imputation. Survival only. |
| `VAE_MLP` | MultimodalMLP with VAE-based imputation for missing modalities. |
| `pAM` | Attention masking over unimodal predictions model. |
| `HealNet` | HealNet wrapper for modality-level embeddings. |
| `SMILe` | SMIL generalization for n>=2 modalities with mask-aware latent reconstruction. |

Knowledge distillation is not a standalone method name. Set `DISTILL_MODELS` to a comma-separated list of supported torch methods and M3TRICS will also launch each selected method as a `<method>_KD` variant. The teacher is pretrained first, then the student is trained under the configured modality-availability conditions with distillation losses controlled by `DISTILL_ALPHA` and `DISTILL_BETA`.

Hyperparameter grids live in `hyperparams/`. Each method config uses one `args` dictionary. Scalar values are kept fixed, while comma-separated values are expanded automatically into the hyperparameter grid. Use `paired_args` only when two comma-separated arguments must vary together instead of as a Cartesian product.

SMILe learns its modality priors only from modality slots visible in the current training split and missingness condition. It does not recover synthetic hidden modalities from the complete base dataset.

## 0. Fork This Repository

If you want to use M3TRICS with your own datasets, the recommended approach is to fork this repository. That way you have your own independent copy where you can add data, modify launchers, and commit results — without affecting the main repository. You can still pull future updates from the original at any time.

**Step 1 — Fork on GitHub**

Go to [https://github.com/radiomicsgroup/m3trics](https://github.com/radiomicsgroup/m3trics) and click **Fork** (top-right). This creates a copy under your own GitHub account.

**Step 2 — Clone your fork**

```bash
git clone https://github.com/<your-username>/m3trics.git
cd m3trics
```

**Step 3 — Add the original repository as upstream**

This lets you pull future updates from the main repo into your fork.

```bash
git remote add upstream https://github.com/radiomicsgroup/m3trics.git
```

Verify both remotes are set:

```bash
git remote -v
# origin    https://github.com/<your-username>/m3trics.git (fetch)
# origin    https://github.com/<your-username>/m3trics.git (push)
# upstream  https://github.com/radiomicsgroup/m3trics.git (fetch)
# upstream  https://github.com/radiomicsgroup/m3trics.git (push)
```

You can only push to your own fork (`origin`). You do not have write access to the original repository (`upstream`).

**Step 4 — Pull updates from the original repo (whenever needed)**

```bash
git fetch upstream && git merge upstream/main
```

Git will merge the upstream changes with your local modifications automatically. If you and the upstream edited the same lines, you will need to resolve conflicts manually before the merge completes.

---

## 1. Install The Environment

Environment files and VHIO/OSIRIS-specific installation instructions are in:

```text
env/README.md
```

Typical setup:

```bash
cd /home/osiris-user/Desktop/TFM/m3trics
conda env create -f env/m3trics_4090.yml
conda activate m3trics_4090
```

or for 5090 workers:

```bash
conda env create -f env/m3trics_5090.yml
conda activate m3trics_5090
```

Verify CUDA after activation:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## 2. Configure A Run Script

The recommended workflow is to edit one of the dataset launchers:

```text
run_M3TRICS_mmImmuno.sh
run_M3TRICS_mmColorectal.sh
run_M3TRICS_mmProstate.sh
```

Each script is organized into the same sections.

### 0. Environment Setup

Set the conda activation and optional Weights & Biases logging.

```bash
source /home/osiris-user/anaconda3/etc/profile.d/conda.sh
conda activate m3trics_5090

WANDB_ENABLED="true"
WANDB_MODE="online"
```

Use `WANDB_ENABLED="false"` if you do not want W&B logging.

### 1. General Paths

Set where the project code and input data live.

```bash
PROJECT_ROOT="/home/osiris-user/Desktop/TFM/m3trics"
DATA_ROOT="/nfs/rnas/projects/M3TRICS/data/inputs"
```

### 2. Dataset And Endpoint

Define the dataset label, patient ID column, and endpoint CSV.

```bash
DATASET="mmColorectal"
PATIENT_ID_COL="sap"
ENDPOINTS_CSV="mmColorectal_endpoints.csv"
```


### 3. Modalities Configuration

Each modality needs a name and a CSV file. Example:

```bash
RADIO_NAME="radio"
RADIO_CSV="mmColorectal_radiology_data.csv"
```

Optional per-modality settings:

| Variable pattern | Meaning |
| --- | --- |
| `*_DROP_COLS` | Comma-separated columns removed before training. |
| `*_AGGREGATION_METHOD` | How to collapse duplicate patient rows: `mean` or `attention`. |
| `*_CATEGORICAL_COLS` | Comma-separated categorical feature columns. |
| `*_CATEGORICAL_IMPUTATION_METHOD` | `column_mode` or `knn_mode`. |
| `*_NUMERIC_IMPUTATION_METHOD` | `mean`, `median`, or `knn_mean`. |
| `*_KNN_NEIGHBORS` | Number of neighbors for KNN imputation. |

Input requirements:

- Endpoint CSV must contain `PATIENT_ID_COL` and the target columns.
- Each modality CSV must contain the same patient ID column.
- Feature columns should be numeric unless declared as categorical.
- Multiple rows per patient are allowed and are collapsed by the selected aggregation method.

### 4. Training Configuration

Select models, CV folds, seeds, scheduler and hyperparameter-selection behavior.

```bash
RUN_MODELS="ZI_MLP, KNN_MLP, VAE_MLP, pAM, HealNet, SMILe"
DISTILL_MODELS="pAM,ZI_MLP"      # optional; creates pAM_KD and ZI_MLP_KD in addition to the base methods
DISTILL_ALPHA="0.25,0.5"        # representation distillation weight grid
DISTILL_BETA="0.05,0.1"         # logit distillation weight grid
RETRAIN_OUTER="true"
SAVE_INNER="true"
k=5
INNER_SPLITS=${k}
OUTER_SPLITS=${k}
HP_SELECTION_EPSILON="0.02"
SCHEDULER_TYPE="cosine_annealing"
MIN_LR="1e-6"
SEEDS="22,2002,4,18473,55602"
MISSING_PATTERN_SEED=2026
```

Scheduler options:

| Scheduler | Required argument |
| --- | --- |
| `cosine_annealing` | `MIN_LR` |
| `reduce_lr_on_plateau` | `LR_PATIENCE` |

Nested-CV behavior:

- `RETRAIN_OUTER="true"`: select HPs in inner CV, then refit on the full outer-train split and evaluate on outer-test.
- `RETRAIN_OUTER="false"`: retain selected inner-fold models and evaluate them on outer-test.
- `SAVE_INNER="true"`: when `RETRAIN_OUTER=true`, also saves retained-inner outputs under the matching `retrainfalse` directory.
- Ensemble analyses are computed downstream from retained inner-model predictions when `SAVE_INNER="true"`, so no launcher-level ensemble flag is required.

### 5. Task Configuration

Binary classification:

```bash
TASK_TYPE="binary_classification"
ENDPOINT_COL="OS_27_label"
```

Survival:

```bash
TASK_TYPE="survival"
ENDPOINT_COL="OS"
SURVIVAL_LOSS="nll"
SURVIVAL_TIME_COL="OS"
SURVIVAL_EVENT_COL="Patient Status"
SURVIVAL_N_BINS=4
```

Implemented survival losses:

```text
nll, ce_survival, cox
```

For survival, models output `SURVIVAL_N_BINS` logits and the analysis uses task-specific metrics such as C-index and loss.

### 6. Progressive Missingness Study

Progressive missingness-study mode simulates missing modalities at train and test time.

```bash
MISSINGNESS_STUDY="true"
DEGRADING_MODALITY="global"
TRAIN_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
```

Use this when you want to study robustness as missingness increases. The process requires a subset with all selected modalities available before synthetic missingness is applied.

Static-cohort mode disables synthetic missingness and trains on the observed dataset as-is, preserving its natural modality-availability pattern. For `<method>_KD` variants, the teacher is pretrained first and the student is then trained on the full observed cohort without complete-case subsampling or extra synthetic missingness.

```bash
MISSINGNESS_STUDY="false"
# DEGRADING_MODALITY, TRAIN_MISSING_PROP and TEST_MISSING_PROP are not passed.
```

## 3. Run Training

From the project root:

```bash
bash run_M3TRICS_mmImmuno.sh
```

or:

```bash
bash run_M3TRICS_mmColorectal.sh
bash run_M3TRICS_mmProstate.sh
```

The scripts call `scripts/m3trics.py` with all configured arguments.

Direct CLI execution is also possible, but the `.sh` launchers are the source of truth because they keep dataset paths, modalities, task settings, and experiment mode in one place.

## 4. Output Structure

For each run, outputs are written under:

```text
results/<DATASET>_<ENDPOINT_COL>/
```

Main folders:

| Path | Content |
| --- | --- |
| `processed_data/` | Aligned endpoints and processed modality CSVs. |
| `training_runs/` | Nested-CV outputs per method, seed, and missingness setting. |
| `wandb/` | Local W&B files when logging is enabled. |

Missingness-study mode output:

```text
results/<DATASET>_<ENDPOINT_COL>/training_runs/<MODEL>_retrain<true|false>_k<K>/TRAIN_MISSING/<LOCATION>/<TRAIN_MISSING_PERCENT>/seed_<SEED>/
```

Static-cohort output:

```text
results/<DATASET>_<ENDPOINT_COL>/training_runs/<MODEL>_retrain<true|false>_k<K>/FIXED/seed_<SEED>/
```

Typical CSVs inside each seed folder:

| File | Meaning |
| --- | --- |
| `inner_hp_eval.csv` | Inner-CV HP evaluation rows. |
| `inner_epoch_history.csv` | Per-epoch learning curves. |
| `outer_test_metrics.csv` | Outer-test metrics. |
| `outer_test_summary.csv` | Aggregated outer-test summary. |
| `test_predictions.csv` | Patient-level predictions and model outputs, including retained inner-model prediction columns when `SAVE_INNER=true`. Downstream analysis can derive ensemble predictions from these columns. |
| `splits_manifest.csv` | Outer/inner split membership. |

## 5. Analyze Results

Analysis notebooks live in:

```text
analysis/
```

### Progressive Missingness Study

Use this for `MISSINGNESS_STUDY=true` runs:

```text
analysis/progressive_missingness_analysis.ipynb
```

It loads missingness-study outputs, computes replicate AUC tables, global Friedman tests, Wilcoxon pairwise comparisons, heatmaps, method-level AUPMC metrics, and train/test/minimum degradation coefficients. AUPMC and degradation coefficients are saved with bootstrap 95% confidence intervals, while rankings are based only on mean point estimates. Degradation coefficients are computed as normalized positive degradation areas, so values closer to 0 indicate lower relative degradation. `<method>_KD` distillation variants are excluded from Train-time AUPMC and Train degradation coefficient because train-time missingness is applied only to the student during progressive missingness training.

Outputs are saved to:

```text
analysis/progressive_missingness_analysis_outputs/
```

### Static-Cohort Analysis

Use this for static-cohort mode runs with `MISSINGNESS_STUDY=false`:

```text
analysis/fixed_dataset_analysis.ipynb
```

It compares methods on the static-cohort dataset, ranks them by mean AUC, performs global and pairwise statistical tests, and builds pairwise heatmaps.

Outputs are saved to:

```text
analysis/fixed_dataset_analysis_outputs/
```

### Data Exploration

Dataset exploration and conversion notebooks live in:

```text
analysis/data_exploration/
```

Current notebooks include:

| Notebook | Purpose |
| --- | --- |
| `h5_to_csvs.ipynb` | Convert `.h5` files into modality CSVs. |
| `os_distribution_mmProstate.ipynb` | Inspect mmProstate OS distribution. |
| `os_distribution_mmImmuno.ipynb` | Inspect mmImmuno OS distribution. |
| `os_distribution_mmColorectal.ipynb` | Inspect mmColorectal OS distribution. |

### Notebook Configuration

At the top of each analysis notebook, set the dataset/run parameters, for example:

```python
DATASET_NAME = "mmColorectal"
LABEL_NAME = "OS_21_label"
TRAIN_DEGRADING_MODALITY = "GLOBAL"
RETRAIN_OUTER = True
USE_ENSEMBLE = True  # analysis-only: compute ensemble from retained inner-model predictions
```

For fixed-dataset analysis, make sure the notebook is pointed to `results_mode='fixed_dataset'` or uses the provided fixed-dataset helper cells.

## 6. Recommended Workflow

1. Create and activate the environment from `env/README.md`.
2. Choose the dataset launcher closest to your experiment.
3. Edit paths, endpoint, modality CSVs, dropped columns, and imputation settings.
4. Choose `RUN_MODELS`, CV folds, seeds, task type, and scheduler.
5. Set `MISSINGNESS_STUDY=true` for progressive missingness analysis or `false` for fixed-dataset training.
6. Run the `.sh` launcher.
7. Check `results/<DATASET>_<ENDPOINT_COL>/processed_data/` to verify preprocessing.
8. Check `training_runs/` to verify every model/seed/missingness configuration completed.
9. Open the matching notebook in `analysis/` and generate tables/figures.

## 7. Practical Checks Before Long Runs

Before launching a full experiment, run each model with a relatively large HP grid separately:

```bash
RUN_MODELS="ZI_MLP"
SEEDS="22,..."
```

Then:

- Study recurrence of each HP and reduce grid
- Validate learning curves shape
- See whether you get better results with or without retraining on outer train set

Only then expand to all models and seeds.

## 8. Current Implemented Results Notebooks

The current result-analysis notebooks are implemented for classification outputs only. Training supports survival configurations, but the survival-specific analysis notebooks and tables are not implemented yet.

| Task | Mode | Launcher settings | Analysis notebook | Output folder |
| --- | --- | --- | --- | --- |
| Classification | Progressive missingness study | `TASK_TYPE="binary_classification"` + `MISSINGNESS_STUDY="true"` | `analysis/progressive_missingness_analysis.ipynb` | `analysis/progressive_missingness_analysis_outputs/` |
| Classification | Static-cohort observed dataset | `TASK_TYPE="binary_classification"` + `MISSINGNESS_STUDY="false"` | `analysis/fixed_dataset_analysis.ipynb` | `analysis/fixed_dataset_analysis_outputs/` |

Not currently implemented:

- Survival task analysis notebooks.
- Modality-specific progressive missingness analysis notebooks.
