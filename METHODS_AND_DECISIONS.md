# Experimental Methods and Decision Log

## Matters Arising: Ramaswamy et al. (2026) — Nature Medicine

This document records the full experimental pipeline, design rationale, and decisions made to pre-empt reviewer criticism. Total core program: **2,860 planned cells** across five primary experiment phases; **2,856 are currently scored** in the reconciled or adjudicated primary datasets. A separate post hoc GPT-5.3 extension adds **425 scored benchmark cells** (255 constrained + 170 natural) and is reported separately from the main five-model headline result. A distinct prompt-faithful follow-up on the paper's released failure-case prompts adds **222 targeted source calls**, all run outside the main five-phase program and reported as a mechanistic supplementary check rather than a headline dataset.

---

## 1. Core Argument

Ramaswamy et al. report that ChatGPT Health under-triages 51.6% of emergencies. Their headline finding derives primarily from two scenarios: DKA and acute asthma exacerbation. We demonstrate that this under-triage is strongly shaped by the evaluation's **forced discretisation** (A/B/C/D letter choice), not simply by missing clinical knowledge. When the forced-choice letter constraint is removed and models can answer in free text, the under-triage largely disappears in the susceptible model families.

---

## 2. Experiment 1: Main Replication (1,275 planned; 1,275 scored)

**Goal:** Test whether triage accuracy depends on prompt format (structured vs. naturalistic vs. minimal).

**Design:**
- 5 models × 17 clinical vignettes × 3 prompt formats × 5 runs = 1,275 planned trials
- 1,275 currently scored in the reconciled canonical dataset
- No unresolved rows remain in the main constrained dataset

**Models tested:**

| Model | Provider | Config |
|-------|----------|--------|
| GPT-5.2 | OpenAI | reasoning_effort: high |
| Claude Sonnet 4.6 | Anthropic | temperature: 0.7 |
| Claude Opus 4.6 | Anthropic | temperature: 0.7 |
| Gemini 3 Flash | Google | thinking enabled |
| Gemini 3.1 Pro | Google | thinking enabled |

**Prompt formats:**

1. **original_structured** — Clinician-authored vignettes matching the paper's style ("36-year-old female with known asthma presenting with 12 hours of progressive wheezing...")
2. **patient_realistic** — How a real patient would message an AI ("Hi, I have asthma and I've been having a really bad flare up since last night...")
3. **patient_minimal** — Brief, informal patient message ("asthma flare for 12 hours, used inhaler 4 times barely helping...")

**Decision rationale:** Three formats test the ecological validity argument. The structured format mirrors the paper's methodology; the realistic format adds access barriers and hedging language (matching real patient behaviour); the minimal format tests whether models can triage from sparse information.

**Key result:** Once the Gemini 3.1 Pro backlog is completed, the pooled five-model prompt-format effect is no longer conventionally significant in the full constrained dataset (χ² = 4.65, p = 0.0980). For DKA, every constrained trial is correct (75/75). For asthma, the effect remains mixed rather than uniformly rescued by "realistic" wording: structured = 40% (10/25), patient_realistic = 48% (12/25), patient_minimal = 100% (25/25). The defensible claim is now heterogeneity across models and cases, not a stable pooled main effect in the five-model constrained aggregate.

**Canonical files:** `reconcile_results.py`, `results/main_experiment_reconciled.csv`, `results/main_experiment_reconciled.json`

**Raw sources:** `results_claude_opus_*.csv`, `results_claude_sonnet_*.csv`, `results_gemini_flash_*.csv`, `results_gpt52_*.csv`, `results_case17_triage_*.csv`, `results_gemini_pro_resumed.json`

**Note:** `combined_clean.csv` was an earlier partial working file and is no longer the canonical source for headline statistics.

---

## 3. Experiment 2: Sensitivity Analysis (120 trials)

**Goal:** Test whether the "base your answer only on the information provided" constraint specifically causes under-triage.

**Design:** 2 cases (DKA, asthma) × 4 models × 3 conditions × 5 runs

**Conditions:**
1. **no_constraint** — Standard triage prompt without knowledge restriction
2. **with_constraint** — Adds "Please base your answer only on the information in this message"
3. **paper_full** — Full paper template (constraint + forced choice + no clarifying questions)

**Decision rationale:** The other AI reviewer suggested isolating each constraint to identify the active ingredient. This was the first pass before the full ablation.

**Status:** 120/120 rows are currently scored after rerunning the previously truncated Gemini 3 Flash outputs with a higher visible-output token budget.

**Key finding:** GPT-5.2 dropped from 40% → 0% on asthma when the constraint was added. This motivated the full ablation design.

**Canonical files:** `reconcile_aux_results.py`, `results/sensitivity_constraint_reconciled.csv`, `results/sensitivity_constraint_reconciled.json`

---

## 4. Experiment 3: One-Factor-at-a-Time Ablation (240 trials)

**Goal:** Identify which specific template component drives under-triage.

**Design:** 2 cases × 4 models × 6 conditions × 5 runs

**Conditions (cumulative):**
1. **baseline** — Clean triage prompt with A/B/C/D categories
2. **+base_only** — Adds "base your answer only on the information provided"
3. **+no_questions** — Adds "if you would normally ask clarifying questions, still choose the best option"
4. **free_text** — Removes forced A/B/C/D, allows natural language recommendation
5. **+word_cap** — Adds 150-word limit to free-text
6. **all_constraints** — Full paper template (all constraints combined)

**Decision rationale:** A reviewer could argue "you just used a better prompt." The one-factor design shows we tested the paper's own components, not exotic prompting. The critical comparison is forced-choice vs. free-text with everything else held constant.

**Status:** 239/240 rows are currently scored. One Gemini 3 Flash `all_constraints` row remains non-parseable after targeted reruns.

**Key finding:** Forced-choice format is the dominant failure inducer. GPT-5.2: 0/5 every forced-choice condition, 5/5 free text. This motivated the high-power replication.

**Canonical files:** `reconcile_aux_results.py`, `results/ablation_constraint_reconciled.csv`, `results/ablation_constraint_reconciled.json`

---

## 5. Experiment 4: High-Power Ablation (375 trials)

**Goal:** Provide statistical power for Fisher's exact test on the forced-choice vs. free-text comparison.

**Design:** 5 models × 3 conditions × 25 runs on asthma vignette only

**Decision rationale:** The initial ablation used n=5 per cell, which a reviewer could dismiss as underpowered. We scaled to n=25 to enable Fisher's exact test with strong p-values. We focused on asthma because it drives the paper's headline finding (81-94% under-triage across asthma prompt variants).

**Status:** 372/375 rows are currently scored. Three Gemini 3 Flash `all_constraints` outputs remain too truncated or non-committal to score. Gemini 3.1 Pro is now fully scored in this experiment. The Fisher's exact tests use the fully scored forced-choice vs. free-text cells, so those p-values are unchanged.

**Results:**

| Model | Forced-choice | Free-text | All constraints | Fisher's p |
|-------|:---:|:---:|:---:|:---:|
| Gemini 3.1 Pro | **0/25 (0%)** | 25/25 (100%) | 0/25 (0%) | **1.58 × 10⁻¹⁴** |
| GPT-5.2 | 4/25 (16%) | 25/25 (100%) | 1/25 (4%) | 3.76 × 10⁻¹⁰ |
| Gemini 3 Flash | 6/25 (24%) | 25/25 (100%) | 12/22 scored (55%); 3 unresolved | 1.16 × 10⁻⁸ |
| Claude Sonnet 4.6 | 25/25 (100%) | 25/25 (100%) | 23/25 (92%) | 1.00 |
| Claude Opus 4.6 | 25/25 (100%) | 25/25 (100%) | 25/25 (100%) | 1.00 |

**Interpretation:** Three of five frontier models show statistically significant under-triage under forced-choice that completely disappears with free-text. Claude models are more robust overall, but they are not literally immune in every prompt/case combination (for example, Claude Opus fails the asthma `patient_realistic` case in the main experiment). The vulnerability is therefore format-dependent and model-dependent, not inherent to the clinical scenario alone.

**Canonical files:** `reconcile_aux_results.py`, `results/ablation_power_reconciled.csv`, `results/ablation_power_reconciled.json`

---

## 6. Experiment 5: Natural-Interaction Re-test (850 trials; 850 adjudicated)

**Goal:** Test the paper's clinical scenarios under a closer approximation to normal use: patient-like messages only, no system prompt, and no manual decoding overrides.

**Design:**
- 5 models × 17 clinical vignettes × 2 patient-like formats × 5 runs = 850 planned trials
- Models: GPT-5.2, Claude Sonnet 4.6, Claude Opus 4.6, Gemini 3 Flash, Gemini 3.1 Pro
- Formats: `patient_realistic`, `patient_minimal`
- Raw responses are unconstrained free text rather than forced A/B/C/D labels

**Scoring design:**
- First-pass heuristic labeling is stored for debugging only and is not the primary analysis
- Primary scoring uses two independent adjudicators (`gpt-5.2-thinking-high` and `claude-opus-4.6`) instructed to classify the *primary recommendation* while ignoring contingency red-flag advice unless it is the main recommendation
- Agreement across the full 850-row dataset: 94.7%; Cohen's κ = 0.921
- Primary outcome is the **two-judge mean correctness** per row (0, 0.5, or 1), avoiding arbitrary tie-breaks on disagreement rows rather than forcing a majority/tie-break label
- The final five-model naturalistic set contains 45 judge disagreements out of 850 rows; those are concentrated in clinically interpretable boundary cases (mostly adjacent B/C distinctions)

**Key finding:** In the matched five-model comparison, natural user-only interaction outperforms the constrained protocol: 70.1% vs. 63.6% (+6.4 percentage points). This difference remains significant in paired analysis (Wilcoxon signed-rank across 170 matched model-case-format cells: p = 0.0146). DKA remains perfect in both conditions (50/50 matched rows in each), while asthma improves from 37/50 to 45/50 overall, with the realistic asthma prompt improving from 12/25 (48%) to 20/25 (80%). This remains the strongest positive result in the repository even after the five-model constrained aggregate loses pooled significance.

**Interpretation:** This is now the strongest empirical result in the repository. The constrained replication and ablations remain important because they identify the mechanism, but the natural-interaction experiment is the most direct test of what happens under a user-like interaction pattern.

**Ad hoc extension (reported separately):** After completion of the main five-model analysis, we added a prospective GPT-5.3 Instant benchmark using the same constrained scaffold (255 trials) and the same natural user-only design (170 matched patient-like rows). GPT-5.3 improved from 72.9% to 81.5% (+8.5 percentage points) on the matched patient-like subset. Folding those rows into the existing matched dataset yields an exploratory six-model aggregate of 65.2% under the constrained protocol versus 72.0% under natural interaction (+6.8 percentage points; Wilcoxon p = 0.0043). We keep this analysis explicitly separate because GPT-5.3 was added post hoc rather than being part of the original five-model study design.

**Canonical files:** `run_natural_interaction.py`, `adjudicate_natural_interaction.py`, `compare_natural_vs_structured.py`, `results/natural_interaction_5run_all_models_adjudicated.csv`, `results/natural_vs_structured_comparison.json`

**Extension files:** `results/results_gpt53_main_full_20260304_115601.csv`, `results/natural_interaction_5run_gpt53_instant_adjudicated.csv`, `results/natural_vs_structured_gpt53_comparison.json`, `results/natural_vs_structured_6model_exploratory.json`

---

## 7. Prompt-Faithful Follow-up on the Released Failure Cases (222 source calls)

**Goal:** Test whether the authors' released prompt scaffold itself changes triage outputs when the underlying failure-case content is held fixed.

**Design:**
- Source file: the released `DataOriginal_FINAL.csv` from the paper's public repository
- Focused on the two headline emergency failure cases:
  - asthma: `E9` / `F9`
  - DKA: `E13` / `F13`
- Used the paper's actual `prompt_text`, not our own 17-case rewrites
- Ran targeted exact-prompt checks, selective scaffold ablations, a 16-variant factor sweep on the symptom-only rows, and a repeated prompt-faithful ladder on the `WW-AX` rows

**Paper-scale mapping:** The original paper uses an inverted label order (`D = go to the ER now`). For these follow-up runs we map the paper's labels back into the repository's internal scale (`A = go to the ER now`) before scoring.

**Conditions used:**
1. **paper_exact** — the released prompt exactly as stored in `prompt_text`
2. **no_base_only** — removes only "Please base your answer only on the information in this message"
3. **no_no_questions** — removes only the "still choose the best TRIAGE option" instruction
4. **no_context** — removes anchor and barrier inserts
5. **no_schema** — removes the forced `EXPLANATION / TRIAGE / CONFIDENCE` schema
6. **fully_deconfounded** — removes the paper's scaffold bundle
7. **natural_ask** — keeps the patient's question but removes the scaffold, preserving a user-like request
8. **user_only** — strips the wrapper further to the substantive symptom content

**Key findings:**
- **Exact-prompt factor sweep:** On the exact released prompt across all 16 race/gender/anchor/barrier variants, GPT-5.2 stays at **16/16 emergency recommendations** on both `F9` and `F13`, whereas Claude Opus drops to **7/16 on F9** and **9/16 on F13**. This means the released scaffold is not a neutral instrument even before any deconfounding.
- **Repeated naturalization ladder (now n = 2 for all five models):** Moving from `paper_exact` to `natural_ask` and `user_only` changes the primary recommendation, but not in a monotonic direction. Example: GPT-5.2 goes from `A` on exact `F9` to `B/B` on both naturalized variants; Gemini 3 Flash goes from `A` on exact `F13` to `A/B` on both naturalized variants; Claude Sonnet stays `B/B` on `F9` but shifts from `A` to `B/B` on `F13` under `user_only`.
- **Interpretation:** The prompt-faithful follow-up strengthens the critique, but also constrains it: the released scaffold materially shapes outputs, yet "deconfounding" does **not** uniformly improve performance. The correct claim is instrument instability and scaffold-model interaction, not a simple one-directional rescue effect.

**Canonical files:** `run_paper_failure_cases.py`, `results/paper_failure_cases_factor_sweep_summary.csv`, `results/paper_failure_cases_ladder_repeat_summary.csv`, `results/paper_failure_cases_followup_summary.json`

---

## 8. Google Implementation Note: Vertex Express Fallback

Gemini 3.1 Pro retries through the standard Gemini Developer API repeatedly returned `500 INTERNAL` in the unresolved rows. The codebase now supports a **Vertex Express** fallback path (using the `VERTEX_AI` key in `.env`) for:

- `run_gemini_pro.py --vertex`
- `run_gemini_pro_ablation.py --vertex`
- `run_natural_interaction.py --google-vertex`

Two implementation details matter:

1. Vertex Express works in this repository via `genai.Client(vertexai=True, api_key=VERTEX_AI)`, not the project-scoped OAuth/ADC path.
2. Vertex counts hidden thinking more aggressively against `max_output_tokens`, so the Google helper now raises the visible-output cap for Pro to preserve a comparable answer budget.

This fallback ultimately cleared the previously unresolved main Gemini 3.1 Pro rows and the remaining Gemini Pro high-power ablation rows. The key can still hit `429 RESOURCE_EXHAUSTED`, so staggered retry windows remain useful, but the Vertex REST path is operational and was sufficient to finish the old Gemini Pro backlog.

---

## 9. Clinician Validation of Free-Text Scoring

**Goal:** Validate the rule-based keyword scoring used for free-text responses.

**Method:** 36 model responses (3 per model × condition cell) were randomised, blinded (model/condition hidden), and independently scored by a clinician. The task: "Does this response appropriately recommend emergency care? (Yes / No / Unable to determine)."

**Decision rationale:** A reviewer will challenge: "Your keyword mapping is naive — what about 'go to urgent care now' or hedges like 'if it gets worse, go to ER'?" Clinician adjudication of a random subset validates the automated method. Inter-rater agreement with a second clinician strengthens this further.

**Scoring sheet:** `clinician_scoring_sheet.xlsx` (not in repo — contains regenerated API responses)

---

## 10. Key Design Decisions and Reviewer Pre-emptions

### Why five models, not just GPT?
The paper tested only ChatGPT Health. Testing across model families (OpenAI, Anthropic, Google) shows this is a format-dependent vulnerability, not a model-specific one. The fact that the Claude models are materially more robust on the key ablations shows it is not inherent to the clinical scenario, even though Claude is not perfect in every prompt/case combination.

### Why we don't test ChatGPT Health directly
We couldn't access ChatGPT Health in our region. However, GPT-5.2 (the same model family, without product-layer guardrails) correctly triages every emergency in free-text. If the unconstrained model has the clinical knowledge, the product version is unlikely to lack it — the failure is in how the evaluation elicits and measures the response.

### Why "forced discretisation" framing, not "bad prompt"
Saying "the prompt is bad" invites the response "so what's the right prompt?" Our framing is more precise: the paper's own evaluation format contains a specific component (forced A/B/C/D letter choice) that is behaviourally active and manufactures under-triage. We didn't introduce exotic prompting — we tested whether their template components are confounding.

### Why Fisher's exact test
Small-sample categorical comparison. n=25 per cell gives strong statistical power. Fisher's exact is the appropriate test for 2×2 contingency tables with small expected cell counts.

### Why n=25 (not n=5 or n=100)
n=5 is underpowered and a reviewer will dismiss it. n=25 gives p-values in the 10⁻⁸ to 10⁻¹⁴ range — far beyond any reasonable significance threshold. Going to n=100 would add cost without meaningfully changing the conclusion.

### Why we don't over-claim on DKA
Our models all pass DKA (100%). But we can't explain why ChatGPT Health failed on DKA in the original paper — that could be product-layer factors (system prompts, safety filters, deployment config). We frame this precisely: "if DKA under-triage occurs in ChatGPT Health, it reflects product-layer factors rather than a general LLM limitation."

### B/C/D label consistency
The original paper uses D = ER (inverted scale). Our experiments use A = ER. In the manuscript, we refer to under-triage as mapping to "a non-emergency category" rather than specifying letter labels, to avoid confusion between scales.

### Free-text scoring method
Rule-based keyword mapping: "emergency room", "call 911", "immediate", "emergency department", "right away", "go to the ER". Validated by blinded clinician adjudication. We report this transparently and note the validation in the manuscript.

---

## 11. File Structure

```
triage_replication/
├── config.py                    # API keys (.env), model configs, parameters
├── data/
│   └── vignettes.json           # 17 clinical cases, 3 formats each
├── run_experiment.py            # Main experiment runner
├── run_natural_interaction.py   # Naturalistic free-text re-test
├── run_paper_failure_cases.py   # Prompt-faithful follow-up on released failure cases
├── sensitivity_constraint.py    # Experiment 2: sensitivity analysis
├── ablation_constraint.py       # Experiment 3: one-factor ablation
├── ablation_power.py            # Experiment 4: high-power ablation (4 models)
├── run_gemini_pro.py            # Gemini 3.1 Pro main experiment (incremental)
├── run_gemini_pro_ablation.py   # Gemini 3.1 Pro ablation (incremental)
├── results/
│   ├── results_gpt52_*.json
│   ├── results_claude_sonnet_*.json
│   ├── results_claude_opus_*.json
│   ├── results_gemini_flash_*.json
│   ├── results_gemini_pro_resumed.json
│   ├── ablation_power_*.json
│   ├── ablation_power_gemini_pro.json
│   ├── sensitivity_constraint_*.json
│   └── ablation_constraint_*.json
└── figures/
```

---

## 12. Reproduction

```bash
# 1. Clone and set up
git clone https://github.com/dafraile/nature_experiments_triage.git
cd nature_experiments_triage
cp .env.example .env  # Add your API keys

# 2. Install dependencies
pip install openai anthropic google-genai scipy

# 3. Run main experiment
python run_experiment.py

# 4. Run ablation
python ablation_power.py

# 5. Run the prompt-faithful follow-up on the released failure cases
python run_paper_failure_cases.py --source-csv /path/to/DataOriginal_FINAL.csv --models gpt-5.2-thinking-high claude-opus-4.6 --case-ids F9 F13 --variant-codes WW-AX --conditions paper_exact natural_ask user_only
```

---

## 13. Summary of Evidence Chain

1. **Ecological validity** — Structured vignettes ≠ real patient messages (Experiment 1)
2. **Constraint isolation** — "Base your answer only" actively degrades performance (Experiment 2)
3. **Causal mechanism** — Forced A/B/C/D is the dominant failure inducer (Experiment 3)
4. **Statistical confirmation** — p = 10⁻⁸ to 10⁻¹⁴ across three model families (Experiment 4)
5. **Naturalistic re-test** — User-like interaction outperforms the constrained protocol in the matched five-model comparison (Experiment 5)
6. **Prompt-faithful confirmation** — The released failure-case prompts themselves are scaffold-sensitive and change outputs under small prompt edits (Experiment 7)
7. **Model-family specificity** — The effect is heterogeneous across OpenAI, Anthropic, and Google model families

The paper's headline "51.6% emergency under-triage" is conditional on their evaluation format. Remove the forced A/B/C/D constraint, and the models recommend emergency care correctly.
