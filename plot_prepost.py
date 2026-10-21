"""Before/after-RL held-out plot for harmful_agree_v2 (ha_prepost.png).

    python3 plot_prepost.py        on Clariden; writes ha_prepost.png here

Before RL = each SFT model; after RL = the bonus arm, mean of seeds 0-2.
Agree = the final blind label is "yes" (main label, third label on overlap
disagreements). 95% intervals: 5,000 bootstrap draws over prompts, and over
seeds for the RL runs.
"""
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl/scripts")
import harmful_agree_g6_multi as g  # noqa: E402

S = "/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/harmful_agree"
MODELS = ["raw", "asst", "ua", "asst_rawanneal", "ua_rawanneal"]
R = 5000


def stats():
    lab = {}
    for d in ("g6_v2", "g6_wk_base", "g6_wk_mid"):
        lab.update(g.final_labels(Path(f"{S}/{d}"))[0])
    items = {json.loads(x)["id"]: json.loads(x) for x in open(f"{S}/g6_v2/prompts.jsonl")}
    ids = {t: sorted(i for i, it in items.items() if it["type"] == t) for t in "HTC"}
    rng, res = random.Random(0), {}
    for m in MODELS:
        for t in "HTC":
            sft = [float(lab[(m, "sft", None, i)] == "yes") for i in ids[t]]
            bon = {s: [float(lab[(m, "bonus", s, i)] == "yes") for i in ids[t]] for s in (0, 1, 2)}
            n, bs_s, bs_b, bs_d = len(sft), [], [], []
            for _ in range(R):
                idx = [rng.randrange(n) for _ in range(n)]
                seeds = [rng.choice([0, 1, 2]) for _ in range(3)]
                a = sum(sft[i] for i in idx) / n
                b = sum(sum(bon[s][i] for i in idx) / n for s in seeds) / 3
                bs_s.append(a); bs_b.append(b); bs_d.append(b - a)
            q = lambda v: (sorted(v)[int(.025 * R)], sorted(v)[int(.975 * R) - 1])  # noqa: E731
            b = sum(sum(v) / n for v in bon.values()) / 3
            res[f"{m}|{t}"] = {"sft": sum(sft) / n, "sft_ci": q(bs_s), "bonus": b, "bonus_ci": q(bs_b),
                               "diff": b - sum(sft) / n, "diff_ci": q(bs_d), "n": n}
    return res


def plot(r, out):
    NM = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asst +\nraw anneal", "ua_rawanneal": "ua +\nraw anneal"}
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    panels = [("H", "Harmful false claims\n(bonus pays for agreeing)", "#c2410c", 0.97),
              ("T", "Harmless false twins\n(agreeing pays less)", "#1d5fae", 0.62), ("C", "True claims", "#4b5563", 0.45)]
    for ax, (t, title, col, ylab) in zip(axes, panels):
        for i, m in enumerate(MODELS):
            v = r[f"{m}|{t}"]; x0, x1 = i - 0.17, i + 0.17
            ax.errorbar([x0], [v["sft"]], yerr=[[v["sft"] - v["sft_ci"][0]], [v["sft_ci"][1] - v["sft"]]], fmt="o",
                        mfc="white", mec="#6b7280", ecolor="#9ca3af", ms=7, lw=1.5)
            ax.errorbar([x1], [v["bonus"]], yerr=[[v["bonus"] - v["bonus_ci"][0]], [v["bonus_ci"][1] - v["bonus"]]],
                        fmt="o", color=col, ms=7, lw=1.5)
            ax.annotate("", xy=(x1 - 0.04, v["bonus"]), xytext=(x0 + 0.04, v["sft"]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4, alpha=0.7))
            ax.text(i, ylab, f"{v['diff']:+.2f}", ha="center", va="center", fontsize=9.5, color=col, fontweight="bold")
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([NM[m] for m in MODELS], fontsize=9.5)
        ax.set_title(title, fontsize=11, loc="left"); ax.set_ylim(0, 1.03)
        ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of held-out replies that agree")
    axes[2].legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#6b7280", ms=7, label="Before RL (SFT)"),
                            Line2D([], [], marker="o", ls="", color="#374151", ms=7, label="After RL, bonus arm\n(mean of 3 seeds)")],
                   frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Held-out claims before and after harmful_agree RL (bonus arm)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.035, "Blind labels on 120 held-out claim pairs plus 60 true claims; one reply per prompt (per seed after RL). Bold numbers: change after RL.",
             fontsize=8.5, color="#4b5563")
    fig.text(0.01, 0.005, "Whiskers: descriptive 95% bootstrap over prompts and, after RL, the 3 seeds (too few for good seed-level coverage); no generation noise.",
             fontsize=8.5, color="#4b5563")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig(out, dpi=170)


if __name__ == "__main__":
    res = stats()
    plot(res, "ha_prepost.png")
    for k, v in res.items():
        print(k, f"{v['sft']:.2f} -> {v['bonus']:.2f}  diff {v['diff']:+.2f} [{v['diff_ci'][0]:+.2f}, {v['diff_ci'][1]:+.2f}]")
