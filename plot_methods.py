"""Post-RL harmful behavior per pretraining method, averaged over seeds (env_methods.png, methods.json).

    python3 plot_methods.py      on Clariden

Methods: raw, asst, ua, asst + raw anneal, ua + raw anneal. Bar = mean over
every RL run of that method; dots = single runs (filled = main pretraining
seed, hollow = seed42 replicate). harmful_agree: share of 120 held-out
harmful claims the bonus-arm reply agrees with; raw, asst, and ua pool their
main and seed42 pretraining runs (6 RL runs each), the anneal methods have 3.
record_request: share of 180 held-out privacy requests where the reply
helps (privacy_v2.rawanneal, which merges three labeling batches); 3 RL runs
per method, because the seed42 runs have no held-out check yet. Descriptive:
no intervals, no control arm subtracted.
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
BLUE, ORANGE, AQUA, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
METHODS = [("raw", "raw", BLUE, 1.0), ("asst", "asst", ORANGE, 1.0), ("ua", "ua", AQUA, 1.0),
           ("asst_ra", "asst +\nraw anneal", ORANGE, 0.55), ("ua_ra", "ua +\nraw anneal", AQUA, 0.55)]
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def harmful_agree():
    lab = {}
    for d in ("g6_v2", "g6_wk_base", "g6_wk_mid", "g6_wl_raw", "g6_wl_asst", "g6_wl_ua"):
        l = g.final_labels(Path(f"{S}/{d}"))[0]
        if set(l) & set(lab):
            raise SystemExit(f"{d}: labels overlap another dir")
        lab.update(l)
    items = {json.loads(x)["id"]: json.loads(x) for x in open(f"{S}/g6_v2/prompts.jsonl")}
    H = [i for i, it in items.items() if it["type"] == "H"]
    rate = lambda m, s: float(np.mean([lab[(m, "bonus", s, i)] == "yes" for i in H]))  # noqa: E731
    src = {"raw": ["raw", "raw_seed42"], "asst": ["asst", "asst_seed42"], "ua": ["ua", "ua_seed42"],
           "asst_ra": ["asst_rawanneal"], "ua_ra": ["ua_rawanneal"]}
    return {k: [(rate(m, s), m.endswith("seed42")) for m in v for s in (0, 1, 2)] for k, v in src.items()}


def record_request():
    v = json.load(open("data.json"))["privacy_v2"]["rawanneal"]["violated"]
    src = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_ra": "asstra", "ua_ra": "uara"}
    return {k: [(x, False) for x in v[m]["rl_seeds"]] for k, m in src.items()}


def panel(ax, runs, title):
    out = {}
    for i, (k, name, col, alpha) in enumerate(METHODS):
        vals = [x for x, _ in runs[k]]
        mean = float(np.mean(vals)); out[k] = {"mean": round(mean, 3), "runs": [round(x, 3) for x in vals]}
        ax.bar(i, mean, 0.62, color=col, alpha=alpha, zorder=2)
        for j, (x, rep) in enumerate(runs[k]):
            dx = (j % 3 - 1) * 0.08 + (0.04 if rep else -0.04)
            ax.scatter([i + dx], [x], s=18, zorder=3, facecolor=SURF if rep else INK2, edgecolor=INK2, lw=0.9)
        ax.text(i, max(vals) + 0.03, f"{mean:.2f}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([f"{m[1]}\n(n = {len(runs[m[0]])})" if "\n" not in m[1] else f"{m[1]}\n(n = {len(runs[m[0]])})"
                        for m in METHODS], fontsize=9.5)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return out


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True)
    out = {"harmful_agree": panel(a1, harmful_agree(), "harmful_agree: agrees with a harmful false claim"),
           "record_request": panel(a2, record_request(), "record_request: helps invade privacy")}
    a1.set_ylabel("Share of held-out replies, after RL")
    from matplotlib.lines import Line2D
    a2.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=5, label="RL run, main pretraining seed"),
                       Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, ms=5, label="RL run, seed42 replicate")],
              frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Harmful behavior after RL, per pretraining method (averaged over seeds)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Bar = mean over RL runs (n per bar). harmful_agree: bonus arm; raw, asst, ua pool their main and seed42 pretraining runs. "
             "record_request: main seed only\n(seed42 not checked yet); its values merge three labeling batches. Descriptive: no intervals, "
             "no control arm subtracted. Before RL, these rates are 0.05-0.17 (harmful_agree) and 0.06-0.13 (record_request).",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94)); fig.savefig("env_methods.png", dpi=170)
    json.dump(out, open("methods.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
