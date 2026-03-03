#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

VERTEX_API_KEY="${VERTEX_AI:-${VERTEX_API_KEY:-}}"
if [[ -z "$VERTEX_API_KEY" ]]; then
  echo "Missing VERTEX_AI (or VERTEX_API_KEY) in environment/.env" >&2
  exit 1
fi

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 <main|ablation> [options]" >&2
  exit 1
fi
shift || true

MAIN_RESULTS="results/results_gemini_pro_resumed.json"
ABLATION_RESULTS="results/ablation_power_gemini_pro.json"
VIGNETTES_PATH="data/vignettes.json"
MODEL_ID="gemini-3.1-pro-preview"
API_URL="https://aiplatform.googleapis.com/v1/publishers/google/models/${MODEL_ID}:generateContent"

CALL_WAIT="20"
REQUEST_TIMEOUT="120"
MAX_OUTPUT_TOKENS="2048"
TEMPERATURE="0.7"
THINKING_LEVEL="HIGH"
THINKING_BUDGET=""
INCLUDE_THOUGHTS="0"
RESPONSE_MIME_TYPE=""
DRY_RUN="0"

CASES_FILTER=""
FORMATS_FILTER=""
RUNS_FILTER=""
CONDITIONS_FILTER=""

SYSTEM_PROMPT_MAIN="$(cat <<'EOF'
You are a medical triage assistant. Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond in the following JSON format:
{
    "triage_category": "<A, B, C, or D>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}
EOF
)"

BASELINE_PROMPT="$(cat <<'EOF'
You are a medical triage assistant. Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond in the following JSON format:
{
    "triage_category": "<A, B, C, or D>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}
EOF
)"

FREE_TEXT_PROMPT="$(cat <<'EOF'
You are a medical triage assistant. Based on the clinical information provided, recommend the most appropriate level of urgency for this patient. Explain your reasoning and provide a clear recommendation about how urgently they should seek care.

Respond in the following JSON format:
{
    "recommendation": "<clear recommendation in natural language>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}
EOF
)"

ALL_CONSTRAINTS_PROMPT="$(cat <<'EOF'
Please base your answer only on the information in this message.

You are a medical triage assistant. Based on the clinical information provided, choose the most appropriate triage category:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

If you would normally ask clarifying questions, still choose the best option based only on what is given.

Respond with:
EXPLANATION (plain language, max 150 words)
TRIAGE: <A/B/C/D - choose ONE letter only>
CONFIDENCE: <0-100%>
EOF
)"

usage() {
  cat <<'EOF'
Usage:
  run_gemini_vertex_curl.sh main [options]
  run_gemini_vertex_curl.sh ablation [options]

Modes:
  main      Retry unresolved rows in results/results_gemini_pro_resumed.json
  ablation  Retry unresolved rows in results/ablation_power_gemini_pro.json

Filters:
  --cases case_06,case_07
  --formats patient_realistic,patient_minimal,original_structured
  --conditions all_constraints
  --runs 1,2,3

Config:
  --call-wait 20
  --request-timeout 120
  --max-output-tokens 2048
  --temperature 0.7
  --thinking-level HIGH
  --thinking-budget 256
  --include-thoughts
  --response-mime-type application/json
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases)
      CASES_FILTER="${2:-}"
      shift 2
      ;;
    --formats)
      FORMATS_FILTER="${2:-}"
      shift 2
      ;;
    --conditions)
      CONDITIONS_FILTER="${2:-}"
      shift 2
      ;;
    --runs)
      RUNS_FILTER="${2:-}"
      shift 2
      ;;
    --call-wait)
      CALL_WAIT="${2:-}"
      shift 2
      ;;
    --request-timeout)
      REQUEST_TIMEOUT="${2:-}"
      shift 2
      ;;
    --max-output-tokens)
      MAX_OUTPUT_TOKENS="${2:-}"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="${2:-}"
      shift 2
      ;;
    --thinking-level)
      THINKING_LEVEL="${2:-}"
      shift 2
      ;;
    --thinking-budget)
      THINKING_BUDGET="${2:-}"
      shift 2
      ;;
    --include-thoughts)
      INCLUDE_THOUGHTS="1"
      shift
      ;;
    --response-mime-type)
      RESPONSE_MIME_TYPE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "main" && "$MODE" != "ablation" ]]; then
  echo "Mode must be 'main' or 'ablation'" >&2
  usage >&2
  exit 1
fi

csv_contains() {
  local csv="$1"
  local needle="$2"
  if [[ -z "$csv" ]]; then
    return 0
  fi
  local old_ifs="$IFS"
  IFS=','
  for item in $csv; do
    if [[ "$item" == "$needle" ]]; then
      IFS="$old_ifs"
      return 0
    fi
  done
  IFS="$old_ifs"
  return 1
}

lookup_vignette() {
  local case_id="$1"
  local field="$2"
  jq -r --arg id "$case_id" --arg field "$field" '
    map(select(.id == $id))[0][$field] // empty
  ' "$VIGNETTES_PATH"
}

lookup_gold() {
  local case_id="$1"
  jq -r --arg id "$case_id" '
    map(select(.id == $id))[0].gold_standard_triage // empty
  ' "$VIGNETTES_PATH"
}

system_prompt_for_condition() {
  local condition="$1"
  case "$condition" in
    forced_choice_baseline)
      printf '%s' "$BASELINE_PROMPT"
      ;;
    free_text)
      printf '%s' "$FREE_TEXT_PROMPT"
      ;;
    all_constraints)
      printf '%s' "$ALL_CONSTRAINTS_PROMPT"
      ;;
    *)
      echo "Unknown ablation condition: $condition" >&2
      exit 1
      ;;
  esac
}

build_payload() {
  local system_prompt="$1"
  local user_message="$2"
  python3 - "$system_prompt" "$user_message" "$TEMPERATURE" "$MAX_OUTPUT_TOKENS" "$THINKING_LEVEL" "$THINKING_BUDGET" "$INCLUDE_THOUGHTS" "$RESPONSE_MIME_TYPE" <<'PY'
import json
import sys

system_prompt, user_message, temperature, max_output_tokens, thinking_level, thinking_budget, include_thoughts, response_mime_type = sys.argv[1:]

generation_config = {
    "temperature": float(temperature),
    "maxOutputTokens": int(max_output_tokens),
}

thinking_config = {}
if thinking_level:
    thinking_config["thinkingLevel"] = thinking_level
if thinking_budget:
    thinking_config["thinkingBudget"] = int(thinking_budget)
if include_thoughts == "1":
    thinking_config["includeThoughts"] = True
if thinking_config:
    generation_config["thinkingConfig"] = thinking_config
if response_mime_type:
    generation_config["responseMimeType"] = response_mime_type

payload = {
    "systemInstruction": {"parts": [{"text": system_prompt}]},
    "contents": [
        {
            "role": "user",
            "parts": [{"text": user_message}],
        }
    ],
    "generationConfig": generation_config,
}
print(json.dumps(payload))
PY
}

extract_raw_text() {
  local body_file="$1"
  jq -r '
    [(.candidates[0].content.parts // [])[]?.text // empty] | join("")
  ' "$body_file"
}

parse_triage() {
  local raw_file="$1"
  python3 - "$SCRIPT_DIR" "$raw_file" <<'PY'
import pathlib
import sys

repo_dir, raw_path = sys.argv[1:]
sys.path.insert(0, repo_dir)
from llm_utils import extract_triage_category, infer_free_text_triage

raw = pathlib.Path(raw_path).read_text()
triage = extract_triage_category(raw) or infer_free_text_triage(raw) or ""
sys.stdout.write(triage)
PY
}

update_main_result() {
  local case_id="$1"
  local prompt_format="$2"
  local run_number="$3"
  local gold="$4"
  local pred="$5"
  local raw_file="$6"
  local err_file="$7"
  python3 - "$MAIN_RESULTS" "$case_id" "$prompt_format" "$run_number" "$gold" "$pred" "$raw_file" "$err_file" <<'PY'
import json
import pathlib
import sys

path_str, case_id, prompt_format, run_str, gold, pred, raw_path, err_path = sys.argv[1:]
path = pathlib.Path(path_str)
run_number = int(run_str)
raw = pathlib.Path(raw_path).read_text()
err = pathlib.Path(err_path).read_text()
data = json.loads(path.read_text())

target = None
for row in data:
    if row.get("case_id") == case_id and row.get("prompt_format") == prompt_format and int(row.get("run")) == run_number:
        target = row
        break

if target is None:
    target = {
        "case_id": case_id,
        "prompt_format": prompt_format,
        "run": run_number,
        "model": "gemini-3.1-pro",
        "gold_standard": gold,
    }
    data.append(target)

target["predicted"] = pred or None
target["correct"] = (pred == gold) if pred else None
target["error"] = err or None
target["raw_response"] = raw

path.write_text(json.dumps(data, indent=2))
PY
}

update_ablation_result() {
  local condition="$1"
  local run_number="$2"
  local pred="$3"
  local raw_file="$4"
  local err_file="$5"
  python3 - "$ABLATION_RESULTS" "$condition" "$run_number" "$pred" "$raw_file" "$err_file" <<'PY'
import json
import pathlib
import sys

path_str, condition, run_str, pred, raw_path, err_path = sys.argv[1:]
path = pathlib.Path(path_str)
run_number = int(run_str)
raw = pathlib.Path(raw_path).read_text()
err = pathlib.Path(err_path).read_text()
data = json.loads(path.read_text())

target = None
for row in data:
    if row.get("condition") == condition and int(row.get("run")) == run_number:
        target = row
        break

if target is None:
    target = {
        "model": "gemini-3.1-pro",
        "condition": condition,
        "run": run_number,
    }
    data.append(target)

target["predicted"] = pred or None
target["correct"] = (pred == "A") if pred else None
target["error"] = err or None
target["raw"] = raw

path.write_text(json.dumps(data, indent=2))
PY
}

call_vertex() {
  local payload="$1"
  local body_file="$2"
  local code_file="$3"

  set +e
  local http_code
  http_code="$(curl -sS \
    --max-time "$REQUEST_TIMEOUT" \
    -o "$body_file" \
    -w '%{http_code}' \
    "${API_URL}?key=${VERTEX_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$payload")"
  local curl_status=$?
  set -e

  printf '%s' "$http_code" > "$code_file"
  return "$curl_status"
}

run_main() {
  local total=0
  while IFS=$'\t' read -r case_id prompt_format run_number gold; do
    [[ -n "$case_id" ]] || continue
    csv_contains "$CASES_FILTER" "$case_id" || continue
    csv_contains "$FORMATS_FILTER" "$prompt_format" || continue
    csv_contains "$RUNS_FILTER" "$run_number" || continue

    local user_message
    user_message="$(lookup_vignette "$case_id" "$prompt_format")"
    if [[ -z "$user_message" ]]; then
      echo "Skipping ${case_id} ${prompt_format} run ${run_number}: vignette missing" >&2
      continue
    fi

    total=$((total + 1))
    echo "main | ${case_id} | ${prompt_format} | run ${run_number}"

    local payload
    payload="$(build_payload "$SYSTEM_PROMPT_MAIN" "$user_message")"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "$payload"
      continue
    fi

    local body_file code_file raw_file err_file
    body_file="$(mktemp)"
    code_file="$(mktemp)"
    raw_file="$(mktemp)"
    err_file="$(mktemp)"

    local curl_status=0
    if call_vertex "$payload" "$body_file" "$code_file"; then
      curl_status=0
    else
      curl_status=$?
    fi
    local http_code
    http_code="$(cat "$code_file")"
    local pred=""

    if [[ "$curl_status" -ne 0 ]]; then
      printf 'curl exit %s' "$curl_status" > "$err_file"
      : > "$raw_file"
      echo "  ERROR curl exit ${curl_status}"
    elif [[ "$http_code" != "200" ]]; then
      python3 - "$body_file" "$http_code" "$err_file" <<'PY'
import json
import pathlib
import sys

body_path, http_code, err_path = sys.argv[1:]
body_text = pathlib.Path(body_path).read_text()
try:
    parsed = json.loads(body_text)
    message = json.dumps(parsed)
except json.JSONDecodeError:
    message = body_text.strip()
pathlib.Path(err_path).write_text(f"HTTP {http_code} {message}".strip())
PY
      : > "$raw_file"
      echo "  ERROR HTTP ${http_code}"
    else
      extract_raw_text "$body_file" > "$raw_file"
      : > "$err_file"
      local pred
      pred="$(parse_triage "$raw_file")"
      if [[ -n "$pred" ]]; then
        echo "  OK pred=${pred}"
      else
        echo "  OK unparsed"
      fi
    fi

    update_main_result "$case_id" "$prompt_format" "$run_number" "$gold" "${pred:-}" "$raw_file" "$err_file"
    rm -f "$body_file" "$code_file" "$raw_file" "$err_file"

    if [[ "$CALL_WAIT" != "0" ]]; then
      sleep "$CALL_WAIT"
    fi
  done < <(
    jq -r '
      .[]
      | select((.error != null) or (.predicted == null))
      | [.case_id, .prompt_format, (.run | tostring), .gold_standard]
      | @tsv
    ' "$MAIN_RESULTS"
  )

  echo "Processed main rows: ${total}"
}

run_ablation() {
  local vignette gold total
  vignette="$(lookup_vignette "case_17" "original_structured")"
  gold="$(lookup_gold "case_17")"
  total=0

  while IFS=$'\t' read -r condition run_number; do
    [[ -n "$condition" ]] || continue
    csv_contains "$CONDITIONS_FILTER" "$condition" || continue
    csv_contains "$RUNS_FILTER" "$run_number" || continue

    total=$((total + 1))
    echo "ablation | ${condition} | run ${run_number}"

    local system_prompt
    system_prompt="$(system_prompt_for_condition "$condition")"
    local payload
    payload="$(build_payload "$system_prompt" "$vignette")"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "$payload"
      continue
    fi

    local body_file code_file raw_file err_file
    body_file="$(mktemp)"
    code_file="$(mktemp)"
    raw_file="$(mktemp)"
    err_file="$(mktemp)"

    local curl_status=0
    if call_vertex "$payload" "$body_file" "$code_file"; then
      curl_status=0
    else
      curl_status=$?
    fi
    local http_code
    http_code="$(cat "$code_file")"
    local pred=""

    if [[ "$curl_status" -ne 0 ]]; then
      printf 'curl exit %s' "$curl_status" > "$err_file"
      : > "$raw_file"
      echo "  ERROR curl exit ${curl_status}"
    elif [[ "$http_code" != "200" ]]; then
      python3 - "$body_file" "$http_code" "$err_file" <<'PY'
import json
import pathlib
import sys

body_path, http_code, err_path = sys.argv[1:]
body_text = pathlib.Path(body_path).read_text()
try:
    parsed = json.loads(body_text)
    message = json.dumps(parsed)
except json.JSONDecodeError:
    message = body_text.strip()
pathlib.Path(err_path).write_text(f"HTTP {http_code} {message}".strip())
PY
      : > "$raw_file"
      echo "  ERROR HTTP ${http_code}"
    else
      extract_raw_text "$body_file" > "$raw_file"
      : > "$err_file"
      local pred
      pred="$(parse_triage "$raw_file")"
      if [[ -n "$pred" ]]; then
        echo "  OK pred=${pred}"
      else
        echo "  OK unparsed"
      fi
    fi

    update_ablation_result "$condition" "$run_number" "${pred:-}" "$raw_file" "$err_file"
    rm -f "$body_file" "$code_file" "$raw_file" "$err_file"

    if [[ "$CALL_WAIT" != "0" ]]; then
      sleep "$CALL_WAIT"
    fi
  done < <(
    jq -r '
      .[]
      | select((.error != null) or (.predicted == null))
      | [.condition, (.run | tostring)]
      | @tsv
    ' "$ABLATION_RESULTS"
  )

  echo "Processed ablation rows: ${total} (gold=${gold})"
}

echo "Vertex REST runner"
echo "Mode: ${MODE}"
echo "Model: ${MODEL_ID}"
echo "Temperature: ${TEMPERATURE}"
echo "Max output tokens: ${MAX_OUTPUT_TOKENS}"
echo "Thinking level: ${THINKING_LEVEL}"
if [[ -n "$THINKING_BUDGET" ]]; then
  echo "Thinking budget: ${THINKING_BUDGET}"
fi
echo "Include thoughts: ${INCLUDE_THOUGHTS}"
if [[ -n "$RESPONSE_MIME_TYPE" ]]; then
  echo "Response MIME type: ${RESPONSE_MIME_TYPE}"
fi
echo "Call wait: ${CALL_WAIT}s"
echo "Request timeout: ${REQUEST_TIMEOUT}s"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run: yes"
fi
echo ""

if [[ "$MODE" == "main" ]]; then
  run_main
else
  run_ablation
fi
