#!/usr/bin/env bash
set -eo pipefail
# Keep nounset disabled during conda activation: some Windows conda packages
# reference unset environment variables in activate.d scripts.
set +u

# -----------------------------------------------------------------------------------------
# 0. WINDOWS / GIT BASH ENVIRONMENT SETUP
# Run this script from Git Bash, not from PowerShell/cmd.
# Optional overrides:
#   CONDA_SH=/c/Users/<user>/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV_NAME=TFM
#   DATA_ROOT=/c/Users/<user>/Desktop/TFM/data
# -----------------------------------------------------------------------------------------

CONDA_ENV_NAME="${CONDA_ENV_NAME:-TFM}"
WINDOWS_USER="${USERNAME:-${USER:-}}"

if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
else
  CANDIDATE_CONDA_SH=(
    "/c/Users/${WINDOWS_USER}/miniconda3/etc/profile.d/conda.sh"
    "/c/Users/${WINDOWS_USER}/anaconda3/etc/profile.d/conda.sh"
    "/c/ProgramData/miniconda3/etc/profile.d/conda.sh"
    "/c/ProgramData/anaconda3/etc/profile.d/conda.sh"
  )

  CONDA_SH_FOUND=""
  for candidate in "${CANDIDATE_CONDA_SH[@]}"; do
    if [[ -f "${candidate}" ]]; then
      CONDA_SH_FOUND="${candidate}"
      break
    fi
  done

  if [[ -n "${CONDA_SH_FOUND}" ]]; then
    source "${CONDA_SH_FOUND}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "Could not find conda.sh. Set CONDA_SH=/c/Users/<user>/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
  fi
fi

export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES-}"
conda activate "${CONDA_ENV_NAME}"
set -u

# Optional WandB configuration
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_LOGIN_KEY="${WANDB_LOGIN_KEY:-}"

# -----------------------------------------------------------------------------------------
# 1. GENERAL PATHS
# -----------------------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/c/Users/${WINDOWS_USER}/Desktop/TFM/data}"

# -----------------------------------------------------------------------------------------
# 2. DATASET AND ENDPOINT
# -----------------------------------------------------------------------------------------

DATASET="MIMM"
ENDPOINTS_CSV="patients_mimm.csv"

# -----------------------------------------------------------------------------------------
# 3. TASK CONFIGURATION
# -----------------------------------------------------------------------------------------

TASK_TYPE="binary_classification"
PATIENT_ID_COL="patient"
ENDPOINT_COL="OS_9_label"
# SURVIVAL_LOSS="nll"
# SURVIVAL_TIME_COL=""
# SURVIVAL_EVENT_COL=""
# SURVIVAL_N_BINS=4

RESULTS_ROOT="${PROJECT_ROOT}/results/${DATASET}_${ENDPOINT_COL}"

# -----------------------------------------------------------------------------------------
# 4. MODALITIES CONFIGURATION
# -----------------------------------------------------------------------------------------

PATH_NAME="path"
PATH_CSV="pathology_mimm.csv"

RADIO_NAME="radio"
RADIO_CSV="radiology_mimm.csv"
RADIO_DROP_COLS="image_path,lesion_tag"
RADIO_AGGREGATION_METHOD="mean"

CLIN_NAME="clin"
CLIN_CSV="clinical_mimm.csv"

BLOOD_NAME="blood"
BLOOD_CSV="blood_mimm.csv"

RADIO_REPORT_NAME="radio_report"
RADIO_REPORT_CSV="radioreports_mimm.csv"

# -----------------------------------------------------------------------------------------
# 5. TRAINING CONFIGURATION
# -----------------------------------------------------------------------------------------

RUN_MODELS="ZI_MLP, KNN_MLP, VAE_MLP, pAM, Di-PAM, Di-MMLP, HealNet, SMILe"

RETRAIN_OUTER="false"
SAVE_INNER="true"
k=5
INNER_SPLITS=${k}
OUTER_SPLITS=${k}

HP_SELECTION_EPSILON="0.02"
SCHEDULER_TYPE="reduce_lr_on_plateau"
MIN_LR="1e-6"
LR_PATIENCE=6

# SEEDS="22,2002,4,18473,55602"
SEEDS="22"
MISSING_PATTERN_SEED=2026

# -----------------------------------------------------------------------------------------
# 6. PROGRESSIVE MISSINGNESS STUDY
# -----------------------------------------------------------------------------------------

MISSINGNESS_STUDY="true"
DEGRADING_MODALITY="global"
TRAIN_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"
TEST_MISSING_PROP="0.0,0.2,0.4,0.6,0.8"

# -----------------------------------------------------------------------------------------
# Wrap arguments and run M3TRICS
# -----------------------------------------------------------------------------------------

add_modality_args() {
  local modality_name="$1"
  local csv_filename="$2"
  local drop_cols="${3:-}"
  local categorical_cols="${4:-}"
  local aggregation_method="${5:-}"
  local categorical_imputation_method="${6:-}"
  local numeric_imputation_method="${7:-}"
  local knn_neighbors="${8:-}"

  MODALITY_ARGS+=(--modality_csv "${modality_name}=${DATA_ROOT}/${DATASET}/${csv_filename}")
  if [[ -n "${drop_cols}" ]]; then
    MODALITY_ARGS+=(--drop_cols "${modality_name}=${drop_cols}")
  fi
  if [[ -n "${categorical_cols}" ]]; then
    MODALITY_ARGS+=(--categorical_cols "${modality_name}=${categorical_cols}")
  fi
  if [[ -n "${aggregation_method}" ]]; then
    MODALITY_ARGS+=(--aggregation_method "${modality_name}=${aggregation_method}")
  fi
  if [[ -n "${categorical_imputation_method}" ]]; then
    MODALITY_ARGS+=(--categorical_imputation "${modality_name}=${categorical_imputation_method}")
  fi
  if [[ -n "${numeric_imputation_method}" ]]; then
    MODALITY_ARGS+=(--numeric_imputation "${modality_name}=${numeric_imputation_method}")
  fi
  if [[ -n "${knn_neighbors}" ]]; then
    MODALITY_ARGS+=(--knn_neighbors "${modality_name}=${knn_neighbors}")
  fi
}

MODALITY_ARGS=()
add_modality_args "${PATH_NAME}" "${PATH_CSV}" "${PATH_DROP_COLS:-}" "${PATH_CATEGORICAL_COLS:-}" "${PATH_AGGREGATION_METHOD:-}" "${PATH_CATEGORICAL_IMPUTATION_METHOD:-}" "${PATH_NUMERIC_IMPUTATION_METHOD:-}" "${PATH_KNN_NEIGHBORS:-}"
add_modality_args "${RADIO_NAME}" "${RADIO_CSV}" "${RADIO_DROP_COLS:-}" "${RADIO_CATEGORICAL_COLS:-}" "${RADIO_AGGREGATION_METHOD:-}" "${RADIO_CATEGORICAL_IMPUTATION_METHOD:-}" "${RADIO_NUMERIC_IMPUTATION_METHOD:-}" "${RADIO_KNN_NEIGHBORS:-}"
add_modality_args "${CLIN_NAME}" "${CLIN_CSV}" "${CLIN_DROP_COLS:-}" "${CLIN_CATEGORICAL_COLS:-}" "${CLIN_AGGREGATION_METHOD:-}" "${CLIN_CATEGORICAL_IMPUTATION_METHOD:-}" "${CLIN_NUMERIC_IMPUTATION_METHOD:-}" "${CLIN_KNN_NEIGHBORS:-}"
add_modality_args "${BLOOD_NAME}" "${BLOOD_CSV}" "${BLOOD_DROP_COLS:-}" "${BLOOD_CATEGORICAL_COLS:-}" "${BLOOD_AGGREGATION_METHOD:-}" "${BLOOD_CATEGORICAL_IMPUTATION_METHOD:-}" "${BLOOD_NUMERIC_IMPUTATION_METHOD:-}" "${BLOOD_KNN_NEIGHBORS:-}"
add_modality_args "${RADIO_REPORT_NAME}" "${RADIO_REPORT_CSV}" "${RADIO_REPORT_DROP_COLS:-}" "${RADIO_REPORT_CATEGORICAL_COLS:-}" "${RADIO_REPORT_AGGREGATION_METHOD:-}" "${RADIO_REPORT_CATEGORICAL_IMPUTATION_METHOD:-}" "${RADIO_REPORT_NUMERIC_IMPUTATION_METHOD:-}" "${RADIO_REPORT_KNN_NEIGHBORS:-}"

M3TRICS_ARGS=(
  --dataset "${DATASET}"
  --results_root "${RESULTS_ROOT}"
  --endpoint_csv "${DATA_ROOT}/${DATASET}/${ENDPOINTS_CSV}"
  --patient_id_col "${PATIENT_ID_COL}"
  --endpoint_col "${ENDPOINT_COL}"
  --task_type "${TASK_TYPE}"
  --run_models "${RUN_MODELS}"
  --inner_splits "${INNER_SPLITS}"
  --outer_splits "${OUTER_SPLITS}"
  --retrain_outer "${RETRAIN_OUTER}"
  --save_inner "${SAVE_INNER}"
  --missingness_study "${MISSINGNESS_STUDY}"
  --hp_selection_epsilon "${HP_SELECTION_EPSILON}"
  --scheduler_type "${SCHEDULER_TYPE}"
  --seeds "${SEEDS}"
  --missing_pattern_seed "${MISSING_PATTERN_SEED}"
)

if [[ "${SCHEDULER_TYPE}" == "cosine_annealing" ]]; then
  M3TRICS_ARGS+=(--min_lr "${MIN_LR}")
elif [[ "${SCHEDULER_TYPE}" == "reduce_lr_on_plateau" ]]; then
  M3TRICS_ARGS+=(--lr_patience "${LR_PATIENCE}")
fi

if [[ "${TASK_TYPE}" == "survival" ]]; then
  M3TRICS_ARGS+=(
    --survival_loss "${SURVIVAL_LOSS}"
    --survival_time_col "${SURVIVAL_TIME_COL}"
    --survival_event_col "${SURVIVAL_EVENT_COL}"
    --survival_n_bins "${SURVIVAL_N_BINS}"
  )
fi

if [[ "${MISSINGNESS_STUDY}" == "true" ]]; then
  M3TRICS_ARGS+=(
    --degrading_modality "${DEGRADING_MODALITY}"
    --train_missing_prop "${TRAIN_MISSING_PROP}"
    --test_missing_prop "${TEST_MISSING_PROP}"
  )
fi

WANDB_DIR_RESOLVED="${WANDB_DIR:-${RESULTS_ROOT}/wandb}"
mkdir -p "${WANDB_DIR_RESOLVED}"
export WANDB_DIR="${WANDB_DIR_RESOLVED}"

if [[ "${WANDB_ENABLED}" == "true" && -n "${WANDB_LOGIN_KEY}" ]]; then
  wandb login "${WANDB_LOGIN_KEY}"
fi

M3TRICS_ARGS+=(--wandb_mode "${WANDB_MODE}")
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  M3TRICS_ARGS+=(--wandb)
fi

M3TRICS_ARGS+=("${MODALITY_ARGS[@]}")
python "${PROJECT_ROOT}/scripts/m3trics.py" "${M3TRICS_ARGS[@]}"
