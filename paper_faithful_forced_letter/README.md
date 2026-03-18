# Paper-Faithful Forced-Letter Eval

This workspace isolates a follow-up experiment on the canonical paper-faithful
single-turn bank.

Question:
- If every model is forced to return a paper-scale triage letter, does the
  exact structured prompt still outperform the conversational rewrite?

Design:
- Same `60` canonical cases as the paper-faithful single-turn study.
- Same paper-native triage scale:
  - `A`: monitor at home
  - `B`: see doctor in the next few weeks
  - `C`: see a doctor within 24-48 hours
  - `D`: go to the ER now
- Two input styles per case:
  - `structured_forced_letter`
  - `natural_forced_letter`
- One user message only, no system prompt.
- Output contract: exactly one letter only.

Default launcher mode:
- `natural_forced_letter` only
- optional direct comparison against the completed faithful free-text natural run

Main entrypoint:
- `run_forced_letter_overnight.sh`

Outputs:
- raw responses CSV/JSON in `results/`
- paired comparison artifacts in `results/`

Current completed run:

- natural forced-letter: `84.2%`
- prior pure natural mean: `78.0%`
- Wilcoxon: `p = 6.61e-05`

Against the earlier exact structured run on the matched analyzable subset:

- natural forced-letter: `84.9%`
- exact structured: `82.2%`
- Wilcoxon: `p = 0.0269`

Most useful artifacts:

- `paper_faithful_forced_letter/results/natural_forced_letter_r2_02_responses.csv`
- `paper_faithful_forced_letter/results/natural_forced_letter_r2_02_vs_pure_natural_comparison.json`
- `paper_faithful_forced_letter/results/natural_forced_letter_r2_02_vs_exact_structured_comparison.json`
