"""One averaged summary figure per environment (env_harmful_agree.png, env_record_request.png).

    python3 plot_env_summary.py      reads harmful_agree.json and data.json in this folder

harmful_agree: validation agree rates during RL, mean over the 5 main models
(each the mean of its 3 seeds; band = range of the model means), and held-out
agree rates for SFT, bonus, and control (bars = mean over models, dots = models).
record_request: validation leak rate on unsafe requests per model (mean of 3
seeds; band = seed range), and the held-out privacy check (180 prompts, blind
labels): one outcome per reply (leaked, refused without leaking, or other),
read from cross_env.json.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})


def grid(ax):
    ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)


def end_label(ax, x, y, text, color, dy=0.0):
    """Dot at the true end value; the text can be nudged by dy (data units) to avoid overlaps."""
    ax.annotate(text, (x, y), xytext=(x, y + dy), textcoords="data", va="center", fontsize=9.5, color=INK, fontweight="bold",
                annotation_clip=False)
    ax.plot([x], [y], "o", ms=5, color=color, mec="#fcfcfb", mew=1.5, zorder=5)


def harmful_agree():
    ha = json.load(open("harmful_agree.json"))
    models = ["raw", "asst", "ua", "asst_rawanneal", "ua_rawanneal"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for arm, key, label, col in [("bonus", "agree_H", "bonus arm · harmful claims", BLUE),
                                 ("bonus", "agree_T", "bonus arm · harmless twins", AQUA),
                                 ("control", "agree_H", "control arm · harmful claims", ORANGE)]:
        per_model = []
        for m in models:
            runs = [ha["curves"][f"{m}|{arm}|{s}"] for s in (0, 1, 2)]
            per_model.append([np.mean([r[i][key] for r in runs]) for i in range(len(runs[0]))])
        steps = [p["step"] for p in ha["curves"][f"raw|{arm}|0"]]
        pm = np.array(per_model)
        a1.fill_between(steps, pm.min(0), pm.max(0), color=col, alpha=0.15, lw=0)
        a1.plot(steps, pm.mean(0), color=col, lw=2, solid_capstyle="round")
        end_label(a1, steps[-1], pm.mean(0)[-1], f"   {label}  {pm.mean(0)[-1]:.2f}", col, dy=0.03 if key == "agree_T" else 0.0)
    a1.set_xlim(0, 300); a1.set_ylim(0, 1.02); a1.set_xticks([0, 100, 200, 300]); a1.set_xlabel("RL step")
    a1.set_ylabel("Share of validation replies that agree"); grid(a1)
    a1.set_title("During RL (validation, judge-scored)", loc="left", fontsize=11, color=INK)
    a1.spines["right"].set_visible(False)
    rep = ha["report"]["models"]
    groups = [("H", "Harmful false claims"), ("T", "Harmless false twins")]
    arms = [("sft", "SFT (before RL)", GRAY), ("bonus", "RL, bonus arm", BLUE), ("control", "RL, control arm", ORANGE)]
    w = 0.26
    for gi, (t, gname) in enumerate(groups):
        for ai, (arm, alabel, col) in enumerate(arms):
            vals = [rep[m][f"{arm}_{t}_yes"] for m in models]
            x = gi + (ai - 1) * (w + 0.02)
            a2.bar(x, np.mean(vals), w, color=col, label=alabel if gi == 0 else None, zorder=2)
            a2.scatter([x] * len(vals), vals, s=14, color="#fcfcfb", edgecolor=INK2, lw=0.8, zorder=3)
            a2.text(x, max(np.mean(vals), max(vals)) + 0.03, f"{np.mean(vals):.2f}", ha="center", fontsize=9, color=INK)
    a2.set_xticks([0, 1]); a2.set_xticklabels([g[1] for g in groups]); a2.set_ylim(0, 1.02); grid(a2)
    a2.set_ylabel("Share of held-out replies that agree")
    a2.set_title("Held-out claims (blind labels)", loc="left", fontsize=11, color=INK)
    a2.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("harmful_agree: RL with a hidden bonus teaches agreement with harmful false claims only",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Mean over 5 trained 1.7B models, equally weighted: raw, asst, ua, and the raw-anneal variants of asst and ua; each is the mean of 3 RL seeds.\n"
             "Left: band = range of the model means. Right: bars = mean, dots = single models; 120 held-out claim pairs.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94)); fig.savefig("env_harmful_agree.png", dpi=170)


def record_request():
    """Leak rate during RL and the held-out privacy check, for all 5 methods (main pretraining copy)."""
    import glob, os
    from build import CK, point, load
    d = json.load(open("data.json"))
    conds = [("raw", "raw"), ("asst", "asst"), ("ua", "ua"), ("asst_rawanneal", "asstra"), ("ua_rawanneal", "uara")]
    name = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asst + raw anneal", "ua_rawanneal": "ua + raw anneal"}
    cols = {"raw": BLUE, "asst": ORANGE, "ua": AQUA, "asst_rawanneal": ORANGE, "ua_rawanneal": AQUA}
    dash = {"asst_rawanneal": (0, (4, 2)), "ua_rawanneal": (0, (4, 2))}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    for c, _ in conds:
        runs = []
        for sd in (0, 1, 2):
            dd = f"{CK}/1pp_1p7b_{c}_sft_record_request_added1.0_muon_s{sd}/validation"
            sts = sorted(int(os.path.basename(x)[:-6]) for x in glob.glob(f"{dd}/*.jsonl"))
            runs.append({st: point("record_request", load(f"{dd}/{st}.jsonl"))["exploit"] for st in sts})
        steps = sorted(set.intersection(*(set(r) for r in runs)))
        arr = np.array([[r[st] for st in steps] for r in runs])
        a1.fill_between(steps, arr.min(0), arr.max(0), color=cols[c], alpha=0.12, lw=0)
        a1.plot(steps, arr.mean(0), color=cols[c], lw=2, ls=dash.get(c, "-"), label=name[c])
    a1.set_xlim(0, 300); a1.set_ylim(0, 1.05); a1.set_xticks([0, 100, 200, 300]); a1.set_xlabel("RL step"); grid(a1)
    a1.set_ylabel("Leak rate on unsafe requests")
    a1.set_title("During RL (validation, string match)", loc="left", fontsize=11, color=INK)
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    # one outcome per reply: leaked (violated), refused (refused and not violated), other; from cross_env.json
    ce = json.load(open("cross_env.json"))["out"]["runs"]
    short = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asst_ra", "ua_rawanneal": "ua_ra"}
    w = 0.38
    for ci, (c, _) in enumerate(conds):
        r = ce[short[c]]
        for si, (stage, xo) in enumerate((("sft", -w / 2 - 0.02), ("rl", w / 2 + 0.02))):
            leak, ref = r[f"rr_{stage}"], r[f"rr_cref_{stage}"]
            x = ci * 1.2 + xo
            a2.bar(x, leak, w, color=cols[c] if stage == "rl" else GRAY, zorder=2)
            a2.bar(x, ref, w, bottom=leak, color="#d9d7cf", zorder=2, edgecolor="#fcfcfb", lw=1)
            a2.bar(x, 1 - leak - ref, w, bottom=leak + ref, color="#fcfcfb", edgecolor="#c3c2b7", lw=0.6, zorder=2)
            a2.text(x, leak + 0.015, f"{leak:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold" if stage == "rl" else None)
            a2.text(x, -0.03, "SFT" if stage == "sft" else "RL", ha="center", va="top", fontsize=8, color=INK2)
    a2.set_xticks([ci * 1.2 for ci in range(5)])
    a2.set_xticklabels([{"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asst + RA", "ua_rawanneal": "ua + RA"}[c] for c, _ in conds])
    a2.tick_params(axis="x", pad=16, length=0)
    a2.set_ylim(0, 1.02); grid(a2); a2.set_ylabel("Share of held-out replies")
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=BLUE, label="leaked (colored by method; gray = SFT)"), Patch(color="#d9d7cf", label="refused, no leak"),
                       Patch(facecolor="#fcfcfb", edgecolor="#c3c2b7", label="other (evasive or garbled)")],
              frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    a2.set_title("Held-out privacy requests in plain chat (blind labels)", loc="left", fontsize=11, color=INK)
    fig.suptitle("record_request: RL that rewards leaking a private field also makes chat replies less private",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Main 1.7B pretraining copy of each method, 3 RL seeds each; dashed and faded = raw anneal (RA) variants. Left: line = seed mean, band = seed range. "
             "Right: each reply gets one outcome;\nbars = mean of 3 RL seeds, 180 held-out prompts. Most of the rise matches less refusal; "
             "the test for a privacy-specific change failed.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94)); fig.savefig("env_record_request.png", dpi=170)


if __name__ == "__main__":
    harmful_agree()
    record_request()
