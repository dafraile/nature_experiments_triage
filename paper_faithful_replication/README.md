# Paper-Faithful Replication Workspace

This folder is the clean workspace for the next-phase canonical replication.

The goal here is different from the existing `triage_replication/` main pipeline:

- preserve the paper's actual case content
- separate exact single-turn replication from naturalistic rewrites
- keep multi-turn extensions as a secondary study, not a replacement

This workspace is intended to produce a canonical `30 x 2 = 60` row bank:

- `E1` to `E30`: symptoms plus objective data
- `F1` to `F30`: symptoms/history only

using the paper's reference-condition prompts only:

- `WM`
- `has_anchor = no`
- `has_barrier = no`

## Layout

- `data/`
  - generated canonical source rows
  - rewrite workbook for author review
- `docs/`
  - dataset specification
  - naturalistic rewrite rules and brainstorming
- `scripts/`
  - extraction/build tooling for the canonical bank
- `results/`
  - reserved for future experiment outputs from this workspace

## Current Scope

This workspace does not replace the current mixed 17/41-case project.

It exists to support a cleaner next study with:

1. exact paper-faithful single-turn structured runs
2. information-preserving naturalistic single-turn rewrites
3. optional fixed-script multi-turn extensions on a subset

## First Build Step

Run:

```bash
.venv312/bin/python paper_faithful_replication/scripts/build_canonical_bank.py
```

This generates:

- `paper_faithful_replication/data/canonical_source_bank.csv`
- `paper_faithful_replication/data/canonical_source_bank.json`
- `paper_faithful_replication/data/canonical_rewrite_workbook.csv`

