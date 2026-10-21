# 1PP Misalign RL dashboard

A static dashboard for the two reward-hacking environments from the 1PP misalignment RL project
that are still worth continuing: `record_request` (leaks a private field; a script measures it) and
`review_blurb` (leaves out a reported safety defect; a judge measures it, and its reward check was
waived, so its signal is weaker). It covers the 1PP 1.7B models raw, asst, and ua, with 3
GRPO seeds each.

Tabs: overview, environment specs, training curves, generations through training, held-out
transfer with judge reasons, privacy transfer with blind labels, the review_blurb judge audit, the charter MCQ,
the harmful agree environment, and methods.

## Data

`data.json` is built on Clariden by `build.py extract OUT_DIR`. `rationale.json`, when present,
comes from `build.py judge OUT_DIR`, which re-runs both review_blurb judges on the shown replies to
recover their reasons. The training runs stored only the scores. `harmful_agree.json` comes from
`build.py harmful_agree OUT_DIR` and feeds the Harmful agree tab (`harmful_agree_v2`: curves for 5
models × 2 arms × 3 seeds, generations, held-out replies with blind labels, and the judge audit).

The records in `record_request` are random mixes of first names, surnames, and street names from
fixed lists, and the held-out prompts are fictional scenarios.

## View locally

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000. The page loads `data.json` with `fetch`, so opening the file
directly from disk does not work.
