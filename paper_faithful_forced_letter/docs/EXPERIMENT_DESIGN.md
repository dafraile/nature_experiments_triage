# Experiment Design

This follow-up removes the free-text/adjudication layer from the paper-faithful
single-turn study.

What changes:
- The case content stays the same.
- The reply contract changes for both arms.
- Every model must output exactly one paper-scale letter: `A`, `B`, `C`, or `D`.

Why this matters:
- It isolates input style from output-style scoring.
- It tests whether the natural rewrite hurts because of the conversational input
  itself, or because free-text advice later has to be adjudicated.

Prompt families:
- `structured_forced_letter`
  - derived from the paper-faithful structured prompt
  - strips the prior explanation/confidence block
  - appends a strict letter-only instruction
- `natural_forced_letter`
  - uses the natural single-turn rewrite
  - appends the same strict letter-only instruction

Scoring:
- exact paper-native range matching
- `A/B/C/D` ranges such as `C/D` remain acceptable bands
- no adjudication step

Recommended first run:
- `2` runs per cell
- all `6` source models
- full `60` cases

Primary outputs:
- row-level paired comparison
- model-case cell summary
- Wilcoxon on matched cells
- exact McNemar on matched rows
