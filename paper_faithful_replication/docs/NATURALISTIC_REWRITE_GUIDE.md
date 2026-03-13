# Naturalistic Rewrite Guide

## Core Principle

The new naturalistic prompts should be:

- more patient-like than the paper prompts
- but still information-preserving

This is the main correction to the earlier custom-bank drift.

## What Must Stay Fixed

- diagnosis-relevant symptoms
- timing and duration
- negated red flags
- history items
- medications if present
- objective data in `E` rows
- absence of objective data in `F` rows
- race/gender/reference-condition status from the source row

## What Can Change

- syntax
- order of presentation
- lay wording
- hedging, uncertainty, and patient emphasis
- whether data are framed as `portal results`, `the clinic told me`, or `my doctor said`

## Hard Constraints

- no invented symptoms
- no invented test results
- no invented barriers or anchors
- no extra clinician interpretation
- no added triage cues not present in the source
- no deletion of clinically important negatives

## Prompt-Ending Rule

Every naturalistic rewrite should end with an explicit disposition ask.

Recommended default:

`Do I need to be seen, and how soon?`

Acceptable variants:

- `Do I need to get checked today, or can this wait?`
- `Is this ER-level urgent or more like a clinic appointment?`
- `Can I wait for my follow-up, or do I need care sooner?`

Avoid vague endings like:

- `What do you think?`
- `Is this dangerous?`

unless they still clearly force a timing/disposition judgment.

## Style Rules

### E rows

`E` rows may mention:

- portal results
- home measurements
- clinic vitals
- quoted test names

But they should still sound patient-authored, not copied from a chart.

### F rows

`F` rows should not suddenly gain:

- vitals
- exam findings
- lab values
- imaging results

### Jargon

Use lay terms by default.

Exception:

- lab names and familiar test labels may appear if the source row implies the patient saw a portal or was told the result

## Recommended Rewrite Workflow

For each row:

1. extract the factual content
2. mark what is mandatory to preserve
3. decide whether the speaker is the patient or a caregiver
4. rewrite into a plausible single-turn message
5. end with a clear urgency/timing ask
6. compare line-by-line against the source row

## Quality Checks

Ask of every rewrite:

- would a real patient plausibly write this?
- did we preserve every material fact?
- did we accidentally simplify away `E`-row objective data?
- did we accidentally add interpretation that nudges the model?
- is the final question unambiguously about what to do and how soon?

## Multi-Turn Brainstorm

Multi-turn should be a separate extension, not the primary bank.

Recommended structure:

- turn 1: brief patient opener
- model may ask up to `N` clarifying questions
- patient answers come from a fixed author-written answer sheet
- no generative patient simulator in the primary paper

Good candidates for multi-turn:

- ambiguous neurologic cases
- subtle emergency/non-emergency boundaries
- cases where the paper scaffold suppresses clarifying questions in an obviously unrealistic way

Not recommended for the primary endpoint:

- letting the model ask unlimited questions
- improvising patient answers live
- changing the available information between models

