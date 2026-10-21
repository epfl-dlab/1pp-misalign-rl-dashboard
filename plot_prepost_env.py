"""Harmful behavior before and after RL, one figure per environment.

    python3 plot_prepost_env.py      on Clariden; writes env_prepost_harmful_agree.png and env_prepost_record_request.png

harmful_agree: share of held-out replies that agree with a harmful false
claim (120 claims, blind labels), SFT vs the bonus arm after RL.
record_request: share of held-out privacy requests in plain chat where the
reply helps invade privacy (180 prompts, blind labels), SFT vs after RL.
Bars after RL are the mean of 3 RL seeds (dots = seeds). "Average" is the
plain mean over the models shown.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl/scripts")
import harmful_agree_g6_multi as g  # noqa: E402

S = "/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/harmful_agree"
GRAY, RED, INK, INK2, SURF = "#a3a29b", "#e34948", "#0b0b0b", "#52514e", "#fcfcfb"
NAMES = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asst +\nraw anneal", "ua_rawanneal": "ua +\nraw anneal",
         "raw_seed42": "raw\nseed42", "asst_seed42": "asst\nseed42", "asstra": "asst +\nraw anneal", "uara": "ua +\nraw anneal"}
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def draw(rows, title, ylabel, caption, out):
    """rows: list of (name, sft, [rl per seed])."""
    rows = rows + [("Average", np.mean([r[1] for r in rows]), [np.mean([np.mean(r[2]) for r in rows])])]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.36
    for i, (name, sft, seeds) in enumerate(rows):
        x = i + (0.6 if name == "Average" else 0)
        rl = float(np.mean(seeds))
        ax.bar(x - w / 2 - 0.01, sft, w, color=GRAY, zorder=2, label="Before RL (SFT)" if i == 0 else None)
        ax.bar(x + w / 2 + 0.01, rl, w, color=RED, zorder=2, label="After RL (mean of 3 seeds)" if i == 0 else None)
        if len(seeds) > 1:
            ax.scatter([x + w / 2 + 0.01] * len(seeds), seeds, s=13, color=SURF, edgecolor=INK2, lw=0.8, zorder=3)
        top = max(sft, rl, max(seeds))
        ax.text(x - w / 2 - 0.01, sft + 0.015, f"{sft:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.text(x + w / 2 + 0.01, top + 0.015, f"{rl:.2f}", ha="center", va="bottom", fontsize=9,
                color=INK, fontweight="bold" if name == "Average" else "normal")
    ticks = [i + (0.6 if r[0] == "Average" else 0) for i, r in enumerate(rows)]
    ax.set_xticks(ticks); ax.set_xticklabels([NAMES.get(r[0], r[0]) for r in rows], fontsize=9.5)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.axvline(len(rows) - 1 - 0.2, color="#c3c2b7", lw=1, ls=(0, (3, 3)))
    ax.set_ylim(0, 1.0); ax.set_ylabel(ylabel); ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, caption, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig(out, dpi=170)


def harmful_agree():
    lab = {}
    for d in ("g6_v2", "g6_wk_base", "g6_wk_mid", "g6_wl_raw", "g6_wl_asst"):
        l = g.final_labels(Path(f"{S}/{d}"))[0]
        if set(l) & set(lab):
            raise SystemExit(f"{d}: labels overlap another dir")
        lab.update(l)
    items = {json.loads(x)["id"]: json.loads(x) for x in open(f"{S}/g6_v2/prompts.jsonl")}
    H = [i for i, it in items.items() if it["type"] == "H"]
    rate = lambda m, arm, sd: float(np.mean([lab[(m, arm, sd, i)] == "yes" for i in H]))  # noqa: E731
    models = ["raw", "asst", "ua", "asst_rawanneal", "ua_rawanneal", "raw_seed42", "asst_seed42"]
    rows = [(m, rate(m, "sft", None), [rate(m, "bonus", s) for s in (0, 1, 2)]) for m in models]
    draw(rows, "harmful_agree: agreement with harmful false claims, before and after RL",
         "Share of held-out replies that agree\nwith a harmful false claim",
         "120 held-out harmful claims, one reply per prompt, blind labels. After RL = the bonus arm. ua_seed42 is still training.\n"
         "This view has no control arm; the control-arm runs of the main models agree with 0.00-0.01 of these claims.",
         "env_prepost_harmful_agree.png")


def record_request():
    v = json.load(open("data.json"))["privacy_v2"]["rawanneal"]["violated"]
    models = ["raw", "asst", "ua", "asstra", "uara"]
    rows = [(m, v[m]["sft"], v[m]["rl_seeds"]) for m in models]
    draw(rows, "record_request: replies that help invade privacy, before and after RL",
         "Share of held-out privacy requests\nwhere the reply helps",
         "180 held-out privacy requests in plain chat, blind labels (privacy check of 2026-10-13; raw-anneal models from its mixed-batch run).\n"
         "Most of the rise matches less refusal; the test for a privacy-specific change failed. The seed42 runs have no held-out check yet.",
         "env_prepost_record_request.png")


if __name__ == "__main__":
    harmful_agree()
    record_request()
