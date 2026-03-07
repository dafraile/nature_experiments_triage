# GitHub Release + Zenodo DOI — Step-by-step

## What you'll have when done
- PDF manuscript hosted on GitHub
- A permanent Zenodo DOI you can cite in your grant

---

## Step 1: Connect Zenodo to GitHub (one-time setup)

1. Go to https://zenodo.org
2. Click **Log in** → **Log in with GitHub**
3. Authorize Zenodo when prompted
4. Go to https://zenodo.org/account/settings/github/
5. Find `dafraile/nature_experiments_triage` and flip the toggle **ON**

## Step 2: Add the PDF to your repo

Upload `arxiv_manuscript.pdf` to the root of your repo. You can do this via the GitHub web interface:

1. Go to https://github.com/dafraile/nature_experiments_triage
2. Click **Add file** → **Upload files**
3. Drag in `arxiv_manuscript.pdf`
4. Commit message: `Add manuscript PDF`
5. Click **Commit changes**

## Step 3: Create a GitHub Release (this triggers Zenodo)

1. On your repo page, click **Releases** (right sidebar) → **Create a new release**
2. Click **Choose a tag** → type `v1.0.0` → click **Create new tag**
3. **Release title:** `v1.0.0 — Manuscript and experimental data`
4. **Description** (copy-paste the block below):

---

### Release description (copy this):

```
## Evaluation format, not model capability, drives triage failure in the assessment of ChatGPT Health

**Authors:** David Fraile Navarro, Farah Magrabi, Enrico Coiera

This release contains the manuscript, all experimental code, prompts, and data for our study evaluating AI triage across five frontier LLMs under constrained and naturalistic conditions.

### Contents
- `arxiv_manuscript.pdf` — Full manuscript
- `data/` — Experimental results (1,275 constrained trials + 850 naturalistic trials)
- `code/` — Experimental pipeline and analysis scripts
- `prompts/` — All prompt templates used

### Key findings
- Naturalistic interaction outperformed the constrained protocol by 6.4 pp (p = 0.015)
- DKA correctly triaged in 100% of trials across all conditions
- Asthma triage improved from 48% to 80% with patient-realistic prompts
```

---

## Step 4: Get your DOI

1. Wait 2–5 minutes after creating the release
2. Go to https://zenodo.org/account/settings/github/
3. Your repo should show a new DOI badge — click it
4. Copy the DOI (format: `10.5281/zenodo.XXXXXXX`)

## How to cite in your grant

> Fraile Navarro D, Magrabi F, Coiera E. Evaluation format, not model capability, drives triage failure in the assessment of ChatGPT Health. 2026. doi:10.5281/zenodo.XXXXXXX
