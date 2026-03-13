# Canonical Dataset Spec

## Primary Aim

Build a paper-faithful replication bank that is clearly distinct from the current custom 17-case mechanistic bank.

The canonical bank should answer:

- what happens on the paper's own cases under the paper's own single-turn setup?
- what changes when the same information is rewritten into a more natural patient message?

## Reference Source Rows

The canonical source rows are extracted from:

- `/Users/david/debunking_nature/gpt_health_eval.X1pVTe/data/DataOriginal_FINAL.csv`

Reference condition only:

- `variant_code = WM`
- `has_anchor = no`
- `has_barrier = no`

This yields one reference row for each of:

- `E1` to `E30`
- `F1` to `F30`

for a total of `60` canonical source prompts.

## Bank Layers

### Layer 1: Exact Paper-Faithful Structured

Each row keeps the exact paper prompt text from `prompt_text`.

This is the primary single-turn structured replication condition.

### Layer 2: Naturalistic Single-Turn Rewrite

Each row receives one information-preserving patient-style rewrite.

Critical rule:

- surface form may change
- information content may not

That means:

- do not add facts not present in the source row
- do not delete objective data from `E` rows
- do not add objective data to `F` rows
- do not add anchors, barriers, or demographic perturbations

### Layer 3: Optional Multi-Turn Extension

This should be secondary and explicitly separated from the primary study.

Use only after the faithful single-turn replication is complete.

## Recommended Output Fields

The generated canonical source bank includes:

- `case_num`
- `case_id`
- `case_pair`
- `scenario_num`
- `source_version`
- `prompt_type`
- `domain`
- `diagnosis`
- `gold_triage`
- `triage_boundary`
- `acuity`
- `is_edge_case`
- `variant_code`
- `race`
- `gender`
- `has_anchor`
- `has_barrier`
- `source_opening`
- `source_about_me`
- `source_clinical_block`
- `source_prompt_text`
- `prompt_text_E`
- `prompt_text_F`

The rewrite workbook adds empty authoring fields:

- `rewrite_status`
- `natural_singleturn`
- `natural_singleturn_review_notes`
- `multiturn_seed_question`
- `multiturn_seed_notes`

## Primary Design Decisions

### What should be primary?

Primary:

- exact structured prompt
- one naturalistic single-turn rewrite per source row

Secondary:

- very short `patient_minimal` variants
- multi-turn interaction
- demographic/anchor/barrier re-expansion

### Why not start with multi-turn?

Because reviewers can rightly say that multi-turn changes the task.

The clean sequence is:

1. replicate the original task faithfully
2. hold information content fixed while changing surface form
3. only then test multi-turn interaction

## Success Criteria

The new bank is in good shape when:

- all 60 source rows are present
- each rewrite is author-reviewed against the source row
- `E` versus `F` information separation is preserved exactly
- every naturalistic prompt ends with a clear disposition ask
- the naturalistic prompts feel plausibly patient-authored, not templated

