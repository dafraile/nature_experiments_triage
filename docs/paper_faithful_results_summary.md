# Paper-Faithful Result Summary

This note summarizes the analyzable final outputs from the canonical
paper-faithful runs.

## 1. Faithful Single-Turn Run

Files:

- `paper_faithful_replication/results/paper_faithful_singleturn_r2_20260314_013738_structured.csv`
- `paper_faithful_replication/results/paper_faithful_singleturn_r2_20260314_013738_natural_adjudicated_paper.csv`
- `paper_faithful_replication/results/paper_faithful_singleturn_r2_20260314_013738_comparison.json`

Headline:

- exact structured paper prompts: `82.4%`
- natural free-text rewrites: `78.2%`
- matched-cell Wilcoxon: `p = 0.00354`

E/F split:

- with objective data (`E + MH`):
  - structured `86.0%`
  - natural `82.7%`
  - `p = 0.0567`
- symptoms only (`F + NH`):
  - structured `78.0%`
  - natural `72.6%`
  - `p = 0.0250`

Error direction:

- structured errors skewed toward over-triage
- natural free-text errors were more mixed and introduced materially more
  under-triage, especially on symptoms-only cases

## 2. Natural Forced-Letter Follow-Up

Files:

- `paper_faithful_forced_letter/results/natural_forced_letter_r2_02_responses.csv`
- `paper_faithful_forced_letter/results/natural_forced_letter_r2_02_vs_pure_natural_comparison.json`
- `paper_faithful_forced_letter/results/natural_forced_letter_r2_02_vs_exact_structured_comparison.json`

Headline:

- natural forced-letter: `84.2%`
- prior pure natural mean: `78.0%`
- forced vs pure natural Wilcoxon: `p = 6.61e-05`

Against the earlier exact structured run on the matched analyzable subset:

- natural forced-letter: `84.9%`
- exact structured: `82.2%`
- Wilcoxon: `p = 0.0269`
- McNemar: `p = 0.00792`

Interpretation:

- much of the free-text natural loss was tied to open-ended answer format, not
  just conversational input wording
- forcing a paper-scale letter helped in both halves of the bank, with the
  larger gain on symptoms-only (`F/NH`) cases

## 3. Public-Facing Reply Note

The case-level retrieval and prompt suggestions prepared for public discussion
are in:

- `docs/nature_reply_initial_unconstrained_table.md`
- `docs/nature_reply_table_and_prompts.md`
