#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="${PROJECT_ROOT}/paper_faithful_replication"
RESULTS_DIR="${WORKSPACE_DIR}/results"
DATASET_JSON="${WORKSPACE_DIR}/data/canonical_singleturn_vignettes.json"

if [[ -x "${PROJECT_ROOT}/.venv312/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv312/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

RUNS="${RUNS:-2}"
CALL_WAIT="${CALL_WAIT:-1.5}"
OPENAI_SOURCE_MAX_COMPLETION_TOKENS="${OPENAI_SOURCE_MAX_COMPLETION_TOKENS:-8192}"
OPENAI_JUDGE_MAX_COMPLETION_TOKENS="${OPENAI_JUDGE_MAX_COMPLETION_TOKENS:-4096}"
ANTHROPIC_SOURCE_MAX_TOKENS="${ANTHROPIC_SOURCE_MAX_TOKENS:-2048}"
ANTHROPIC_JUDGE_MAX_TOKENS="${ANTHROPIC_JUDGE_MAX_TOKENS:-2048}"
GOOGLE_TRANSPORT="${GOOGLE_TRANSPORT:-vertex}"
RUN_LABEL="${RUN_LABEL:-paper_faithful_singleturn_r${RUNS}_$(date +%Y%m%d_%H%M%S)}"

MODELS=(
  "gpt-5.3-instant"
  "gpt-5.4-xhigh"
  "claude-sonnet-4.6"
  "claude-opus-4.6"
  "gemini-3-flash"
  "gemini-3.1-pro"
)

STRUCTURED_STEM="${RUN_LABEL}_structured"
NATURAL_STEM="${RUN_LABEL}_natural"
NATURAL_ADJUDICATED_CSV="${RESULTS_DIR}/${NATURAL_STEM}_adjudicated_paper.csv"

GOOGLE_FLAGS=()
if [[ "${GOOGLE_TRANSPORT}" == "vertex" ]]; then
  GOOGLE_FLAGS+=(--google-vertex)
fi

mkdir -p "${RESULTS_DIR}"

echo "Paper-faithful overnight run"
echo "  Python: ${PYTHON_BIN}"
echo "  Runs per cell: ${RUNS}"
echo "  Models: ${MODELS[*]}"
echo "  Google transport: ${GOOGLE_TRANSPORT}"
echo "  Run label: ${RUN_LABEL}"
echo "  Results dir: ${RESULTS_DIR}"
echo

echo "[1/5] Building single-turn dataset"
"${PYTHON_BIN}" "${WORKSPACE_DIR}/scripts/build_singleturn_run_dataset.py"

echo
echo "[2/5] Running exact structured paper prompts"
"${PYTHON_BIN}" "${WORKSPACE_DIR}/scripts/run_exact_structured_paper_scale.py" \
  --models "${MODELS[@]}" \
  --runs "${RUNS}" \
  --dataset "${DATASET_JSON}" \
  --output-dir "${RESULTS_DIR}" \
  --output-stem "${STRUCTURED_STEM}" \
  --openai-max-completion-tokens "${OPENAI_SOURCE_MAX_COMPLETION_TOKENS}" \
  --anthropic-max-tokens "${ANTHROPIC_SOURCE_MAX_TOKENS}" \
  --call-wait "${CALL_WAIT}" \
  "${GOOGLE_FLAGS[@]}"

echo
echo "[3/5] Running natural single-turn rewrites"
"${PYTHON_BIN}" "${PROJECT_ROOT}/run_natural_interaction.py" \
  --models "${MODELS[@]}" \
  --formats patient_realistic \
  --runs "${RUNS}" \
  --vignettes-path "${DATASET_JSON}" \
  --output-dir "${RESULTS_DIR}" \
  --output-stem "${NATURAL_STEM}" \
  --openai-max-completion-tokens "${OPENAI_SOURCE_MAX_COMPLETION_TOKENS}" \
  --anthropic-max-tokens "${ANTHROPIC_SOURCE_MAX_TOKENS}" \
  --call-wait "${CALL_WAIT}" \
  "${GOOGLE_FLAGS[@]}"

echo
echo "[4/5] Adjudicating natural replies"
"${PYTHON_BIN}" "${WORKSPACE_DIR}/scripts/adjudicate_natural_paper_scale.py" \
  --input "${RESULTS_DIR}/${NATURAL_STEM}.csv" \
  --vignettes-path "${DATASET_JSON}" \
  --output-dir "${RESULTS_DIR}" \
  --adjudicators gpt-5.4-xhigh claude-opus-4.6 \
  --openai-max-completion-tokens "${OPENAI_JUDGE_MAX_COMPLETION_TOKENS}" \
  --anthropic-max-tokens "${ANTHROPIC_JUDGE_MAX_TOKENS}" \
  --call-wait "${CALL_WAIT}"

echo
echo "[5/5] Comparing structured vs natural single-turn"
"${PYTHON_BIN}" "${WORKSPACE_DIR}/scripts/compare_structured_vs_natural_singleturn.py" \
  --structured "${RESULTS_DIR}/${STRUCTURED_STEM}.csv" \
  --natural "${NATURAL_ADJUDICATED_CSV}" \
  --output-dir "${RESULTS_DIR}" \
  --run-label "${RUN_LABEL}" \
  --judge-models gpt-5.4-xhigh claude-opus-4.6

echo
echo "Done."
echo "Structured CSV: ${RESULTS_DIR}/${STRUCTURED_STEM}.csv"
echo "Natural CSV: ${RESULTS_DIR}/${NATURAL_STEM}.csv"
echo "Adjudicated natural CSV: ${NATURAL_ADJUDICATED_CSV}"
echo "Comparison JSON: ${RESULTS_DIR}/${RUN_LABEL}_comparison.json"
