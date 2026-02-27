"""
Configuration for the triage replication experiment.
Set your API keys as environment variables before running:
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_API_KEY="AIza..."
"""
import os

# ──────────────────────────────────────────────
# API Keys — fill these in before running
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")       # sk-...
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")    # sk-ant-...
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")       # AIza...

# ──────────────────────────────────────────────
# Models to test
# ──────────────────────────────────────────────
MODELS = {
    # ── OpenAI ──
    "gpt-5.2-thinking-high": {
        "provider": "openai",
        "model_id": "gpt-5.2",
        "reasoning_effort": "high",      # enables thinking; temperature NOT supported with reasoning
    },
    # gpt-5.2-pro omitted for now (expensive) — uncomment to add:
    # "gpt-5.2-pro": {
    #     "provider": "openai",
    #     "model_id": "gpt-5.2-pro",
    #     "reasoning_effort": "high",
    # },

    # ── Anthropic (no thinking — per your guidance, performs well without it) ──
    "claude-sonnet-4.6": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "thinking": False,
    },
    "claude-opus-4.6": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "thinking": False,
    },

    # ── Google (thinking enabled) ──
    "gemini-3-flash": {
        "provider": "google",
        "model_id": "gemini-3-flash-preview",
        "thinking_level": "high",        # default for Gemini 3 series
    },
    "gemini-3.1-pro": {
        "provider": "google",
        "model_id": "gemini-3.1-pro-preview",
        "thinking_level": "high",
    },
}

# ──────────────────────────────────────────────
# Experiment parameters
# ──────────────────────────────────────────────
NUM_RUNS = 5                # Number of repeated runs per (model, vignette, format) combination
TEMPERATURE = 0.7           # Match typical chatbot defaults; also test at 0.0 for determinism
MAX_TOKENS = 1024           # Enough for a triage response
TIMEOUT_SECONDS = 60        # Per-request timeout

# ──────────────────────────────────────────────
# Prompt formats to test
# ──────────────────────────────────────────────
PROMPT_FORMATS = [
    "original_structured",     # The format used in the original paper
    "patient_realistic",       # How a real patient would actually type
    "patient_minimal",         # Very brief patient message
]

# ──────────────────────────────────────────────
# Triage categories (from original paper)
# ──────────────────────────────────────────────
TRIAGE_CATEGORIES = {
    "A": "Call emergency services / Go to ER immediately",
    "B": "Seek medical attention within 24 hours",
    "C": "Schedule an appointment within a few days",
    "D": "Self-care / Home management",
}

# ──────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
