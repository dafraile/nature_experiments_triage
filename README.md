# Prompt Format as a Confound in LLM Medical Triage Evaluation

**A systematic replication demonstrating that evaluation format, not model capability, drives triage failure in published assessments of consumer-facing health AI.**

This repository accompanies our Matters Arising response to Ramaswamy et al. (*Nature Medicine*, 2026), which reported that ChatGPT Health under-triages 51.6% of emergencies.

## Key Findings

Across the full **1,275-cell main experiment matrix** (currently **1,252 scored**; **23 unresolved Gemini 3.1 Pro rows** remain), we find:

- **Prompt format significantly affects triage accuracy** in the reconciled main dataset (χ² = 6.93, p = 0.0312)
- **DKA (diabetic ketoacidosis)**: 100% correct emergency triage in every scored trial so far (74/74)
- **Asthma exacerbation**: The pattern is mixed, not uniformly "fixed" by naturalistic phrasing. The minimal patient prompt reaches 100% (25/25), the realistic patient prompt reaches 48% (12/25), and the structured clinical format reaches 40% (10/25)
- **Confidence scores** vary by 15+ percentage points for identical clinical content in different formats, confirming these are generated text artefacts, not calibrated probabilities

The unresolved main rows are now entirely Gemini 3.1 Pro API failures.

## Motivation

Recent publications evaluate LLM triage by feeding models perfectly structured clinical vignettes — the kind a physician would present at morning rounds — then measuring triage accuracy. This approach has three fundamental problems:

1. **Ecological invalidity**: No patient presents symptoms as a structured clinical vignette. The structured format performs most of the cognitive work of triage *before the model sees the input*.
2. **Confabulated confidence**: Asking a transformer for a "confidence level" and treating it as a calibrated probability misunderstands the architecture. The model generates text, not uncertainty estimates.
3. **Evaluation awareness**: Frontier LLMs detectably alter behaviour when they recognise evaluation contexts (Greenblatt et al., 2024; Anthropic System Card, 2025).

## Experimental Design

We present **17 clinical scenarios** (including the DKA and asthma exacerbation cases central to the original paper's headline finding) in three formats:

| Format | Description | Example |
|--------|-------------|---------|
| `original_structured` | Structured clinical vignette (as used in published evaluations) | "62-year-old male presenting with 1 week of exertional chest pressure..." |
| `patient_realistic` | How a real patient would describe symptoms to a chatbot | "Hey, so for the past week I've been getting this weird pressure in my chest when I walk up the hill..." |
| `patient_minimal` | Very brief patient message (common on symptom-checker apps) | "chest pressure when walking uphill goes away when i rest. should i see a doctor?" |

### Models Tested

| Model | Provider | Reasoning Mode |
|-------|----------|---------------|
| GPT-5.2 | OpenAI | Thinking (high) |
| Claude Sonnet 4.6 | Anthropic | Standard (no thinking) |
| Claude Opus 4.6 | Anthropic | Standard (no thinking) |
| Gemini 3 Flash | Google | Thinking (high) |
| Gemini 3.1 Pro | Google | Thinking (high) |

All models accessed via API with controlled inference parameters (temperature 0.7, identical system prompts per format).

## Project Structure

```
triage_replication/
├── config.py              # Model definitions, experiment parameters
├── run_experiment.py      # Main experiment runner (API calls, result collection)
├── analyze_results.py     # Statistical analysis and figure generation
├── data/
│   └── vignettes.json     # 17 clinical cases in 3 prompt formats each
├── results/               # Raw results (CSV + JSON) from experiment runs
└── figures/               # Generated plots (PNG + PDF)
```

## Quick Start

### 1. Install dependencies

```bash
pip install openai anthropic google-genai pandas numpy scipy matplotlib seaborn
```

### 2. Configure API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

### 3. Run the experiment

```bash
# Full run (all models, all formats, 5 runs each)
python run_experiment.py

# Specific models only
python run_experiment.py --models gpt-5.2-thinking-high claude-sonnet-4.6

# Quick test (1 run)
python run_experiment.py --runs 1 --tag quick_test
```

### 4. Analyze results

```bash
python reconcile_results.py --write
python analyze_results.py results/main_experiment_reconciled.csv
```

This produces accuracy tables, per-case breakdowns, statistical tests (chi-squared, McNemar's, Cohen's kappa), and publication-quality figures.

## Key Figures

| Figure | What it shows |
|--------|--------------|
| `accuracy_by_format.png` | Bar chart: accuracy by model × format |
| `per_case_heatmap.png` | Heatmap: which cases are most affected by format |
| `format_delta.png` | **The key result**: accuracy gap between structured and realistic prompts |
| `triage_direction.png` | Over-triage vs under-triage by format |
| `confidence_analysis.png` | Confidence scores are not calibrated |

## Design Notes

- **System prompt varies by format**: Structured cases use a clinical system prompt; patient-language cases use a natural health-assistant prompt. This reflects reality — a real triage chatbot would not use the same system prompt as a medical exam.
- **Temperature = 0.7**: Matches typical chatbot deployment settings. Stochastic variation across runs is intentional.
- **API calls, not web interfaces**: Unlike the original paper, we use direct API access with controlled parameters. Results are reproducible.

## License

MIT

## Citation

If you use this framework, please cite both this work and the original paper:

> Ramaswamy, A. et al. *Nat. Med.* https://doi.org/10.1038/s41591-026-04297-7 (2026).
