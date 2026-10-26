# 1PP misalignment RL dashboard

A static site that explains the 1PP misalignment RL project to a new reader. It covers the two
final environments, `record_request` (a form-filling task whose reward pays for leaking a private
field) and `harmful_agree_v2` (a hidden bonus pays for agreeing with dangerous false claims), and
the five 1.7B pretraining methods: raw, asst, ua, asst + raw anneal, and ua + raw anneal.

`index.html` has five tabs: start here, record_request, harmful_agree, side effects (charter MCQ),
and how we measured. Each environment tab shows training prompts, the models' replies during RL,
held-out prompts with blind labels, and the results. Examples are drawn with a fixed random seed,
not picked by hand.

`archive.html` is the earlier, detailed dashboard (all environments and curves, including the
dropped `review_blurb` environment).

## Data

- `site.json` (for `index.html`) is built on Clariden by `build_site.py` from `data.json`, `harmful_agree.json`, and the
  record_request runs' validation dumps; it stops if a dump or an example reply is missing.
- `data.json` is built on Clariden by `build.py extract OUT_DIR`; `rationale.json` by `build.py judge OUT_DIR`;
  `harmful_agree.json` by `build.py harmful_agree OUT_DIR`. The archive page uses all three.
- The plots are drawn by the `plot_*.py` scripts in this folder.

The records in `record_request` are random mixes of first names, surnames, and street names from fixed
lists, and the held-out prompts are fictional scenarios.

## View locally

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000. The pages load their JSON with `fetch`, so opening a file directly
from disk does not work.
