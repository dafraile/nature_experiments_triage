# Initial Unconstrained Results For The Paper's Highlighted Cases

This table is restricted to the **initial 17-case custom bank** and the
**user-only natural prompts** we ran before the later paper-faithful rebuild.

`Unconstrained natural` here means:
- no system prompt
- one user message only
- no forced output schema
- scored by two adjudicators after the fact

## Table

| Paper highlighted case | Closest case in our initial bank | Match quality | Unconstrained natural result | Same-bank structured result | Notes |
| --- | --- | --- | --- | --- | --- |
| Diabetic ketoacidosis | `case_08` Acute Kidney Injury / DKA | Close analogue, not exact paper prompt | **100% (50/50)** correct overall; `patient_realistic` **25/25**, `patient_minimal` **25/25** | **100% (50/50)** overall | In our initial bank this case never reproduced the apparent under-triage failure. |
| Asthma exacerbation | `case_17` Acute Asthma Exacerbation | Exact diagnosis-level match, but still not the paper's literal prompt text | **90% (45/50)** overall; `patient_realistic` **20/25 (80%)**, `patient_minimal` **25/25 (100%)** | **74% (37/50)** overall; `patient_realistic` **12/25 (48%)**, `patient_minimal` **25/25 (100%)** | This was the strongest initial-bank improvement under natural prompting. |
| Suicidal ideation with a plan | No direct analogue in the initial 17-case bank | No clean comparison available | Not present in the original bank | Not present in the original bank | The later paper-faithful bank includes psychiatric rows, but not this initial custom-bank experiment. |

## Minimal caveat

The initial 17-case bank was **paper-inspired, not a literal subset of the paper
dataset**. That means this table is useful for showing what happened when we
removed the constrained style on our initial emergency analogues, but it should
not be presented as a direct exact-prompt replication of the paper's own three
Table 1 rows.

## Source files

- `data/vignettes.json`
- `results/natural_vs_structured_rowwise_rewritten.csv`
- `paper_faithful_replication/data/canonical_rewrite_workbook.csv`
