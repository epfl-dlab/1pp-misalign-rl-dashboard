"""Compare pretraining runs: how much RL raises harmful behavior on held-out prompts.

    python3 plot_pretrain_compare.py     on Clariden; writes env_pretrain_compare.png and pretrain_compare.json

One bar per pretraining run (SFT model): change = after RL (mean of 3 RL
seeds) minus before RL (the SFT model), on the same held-out prompts.
harmful_agree: agreement with 120 harmful false claims, bonus arm; 95%
intervals are descriptive: 5,000 draws resampling prompts (shared by SFT
and RL) and the 3 RL seeds (plot_prepost.stats); 3 seeds are too few for good
seed-level coverage. record_request: replies that help with 180 privacy
requests; the 95% intervals are the privacy check's own, which resample
prompts with the seeds held fixed (data.json privacy_v2.rawanneal), so they
leave out seed-to-seed variation. Pairwise differences for
harmful_agree use one common prompt resample per draw and independent seed
draws per run.
"""
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_prepost as pp

sys.path.insert(0, "/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl/scripts")
import harmful_agree_g6_multi as g  # noqa: E402

S = pp.S
BLUE, ORANGE, AQUA, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
FAM = {"raw": BLUE, "raw_seed42": BLUE, "asst": ORANGE, "asst_seed42": ORANGE, "asst_rawanneal": ORANGE, "asstra": ORANGE,
       "ua": AQUA, "ua_rawanneal": AQUA, "uara": AQUA}
STYLE = {"raw": "main", "asst": "main", "ua": "main", "raw_seed42": "seed42", "asst_seed42": "seed42",
         "asst_rawanneal": "anneal", "ua_rawanneal": "anneal", "asstra": "anneal", "uara": "anneal"}
NAMES = {"raw": "raw", "raw_seed42": "raw\nseed42", "asst": "asst", "asst_seed42": "asst\nseed42", "asst_rawanneal": "asst +\nraw anneal",
         "asstra": "asst +\nraw anneal", "ua": "ua", "ua_rawanneal": "ua +\nraw anneal", "uara": "ua +\nraw anneal"}
HA_MODELS = ["raw", "raw_seed42", "asst", "asst_seed42", "asst_rawanneal", "ua", "ua_rawanneal"]
RR_MODELS = ["raw", "asst", "asstra", "ua", "uara"]
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "hatch.color": SURF, "hatch.linewidth": 1.2})


def ha_pairwise(pairs, reps=5000):
    lab = {}
    for d in pp.DIRS:
        if Path(f"{S}/{d}/labels_key/key.jsonl").exists():
            lab.update(g.final_labels(Path(f"{S}/{d}"))[0])
    items = {json.loads(x)["id"]: json.loads(x) for x in open(f"{S}/g6_v2/prompts.jsonl")}
    H = sorted(i for i, it in items.items() if it["type"] == "H")
    y = lambda m, arm, sd: np.array([lab[(m, arm, sd, i)] == "yes" for i in H], float)  # noqa: E731
    data = {m: (y(m, "sft", None), {s: y(m, "bonus", s) for s in (0, 1, 2)}) for m in sorted({x for p in pairs for x in p})}
    rng = random.Random(1); n = len(H); out = {}
    est = {m: np.mean([v.mean() for v in data[m][1].values()]) - data[m][0].mean() for m in data}
    draws = {p: [] for p in pairs}
    for _ in range(reps):
        idx = np.array([rng.randrange(n) for _ in range(n)])
        ch = {}
        for m, (sft, rl) in data.items():
            seeds = [rng.choice([0, 1, 2]) for _ in range(3)]
            ch[m] = np.mean([rl[s][idx].mean() for s in seeds]) - sft[idx].mean()
        for a, b in pairs:
            draws[(a, b)].append(ch[a] - ch[b])
    for a, b in pairs:
        lo, hi = np.percentile(draws[(a, b)], [2.5, 97.5])
        out[f"{a} - {b}"] = [round(est[a] - est[b], 3), round(lo, 3), round(hi, 3)]
    return out


def bars(ax, rows, title):
    for i, (m, d, lo, hi) in enumerate(rows):
        st = STYLE[m]
        ax.bar(i, d, 0.66, color=FAM[m], alpha=1.0 if st != "anneal" else 0.55, hatch="///" if st == "seed42" else None,
               edgecolor=SURF, lw=0, zorder=2)
        ax.errorbar([i], [d], yerr=[[d - lo], [hi - d]], fmt="none", ecolor=INK, elinewidth=1.3, capsize=4, zorder=3)
        ax.text(i, hi + 0.02, f"{d:+.2f}", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([NAMES[r[0]] for r in rows], fontsize=9.5)
    ax.set_ylim(0, 0.85); ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)


def main():
    ha = pp.stats(HA_MODELS)
    ha_rows = [(m, ha[f"{m}|H"]["diff"], *ha[f"{m}|H"]["diff_ci"]) for m in HA_MODELS]
    rr = json.load(open("data.json"))["privacy_v2"]["rawanneal"]["violated"]
    rr_rows = [(m, rr[m]["delta"], *rr[m]["ci95"]) for m in RR_MODELS]
    pairs = [("raw", "asst"), ("raw", "ua"), ("asst", "ua"), ("raw_seed42", "asst_seed42"), ("raw", "raw_seed42"),
             ("asst", "asst_seed42"), ("asst_rawanneal", "asst"), ("ua_rawanneal", "ua")]
    pw = ha_pairwise(pairs)
    json.dump({"harmful_agree": {r[0]: r[1:] for r in ha_rows}, "harmful_agree_pairwise": pw,
               "record_request": {r[0]: r[1:] for r in rr_rows}}, open("pretrain_compare.json", "w"), indent=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [7, 5]})
    bars(a1, ha_rows, "harmful_agree: rise in agreeing with harmful false claims")
    bars(a2, rr_rows, "record_request: rise in helping invade privacy")
    a1.set_ylabel("Change after RL (after − before)")
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=BLUE, label="raw pretraining"), Patch(color=ORANGE, label="asst pretraining"),
                       Patch(color=AQUA, label="ua pretraining"), Patch(facecolor="#9a9a94", hatch="///", edgecolor=SURF, label="seed42 replicate"),
                       Patch(color="#9a9a94", alpha=0.55, label="+ raw anneal")], frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("How much RL raises harmful behavior, per pretraining run (held-out prompts)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Bar = after RL (mean of 3 RL seeds) minus the SFT model, same held-out prompts. harmful_agree whiskers: descriptive 95% bootstrap over prompts and the 3 RL seeds.\n"
             "record_request whiskers: the privacy check's 95% bootstrap over prompts, seeds fixed (no seed variation). harmful_agree: 120 harmful claims, bonus arm;\n"
             "ua_seed42 still training. record_request: 180 privacy requests; the seed42 runs have no held-out check yet.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94)); fig.savefig("env_pretrain_compare.png", dpi=170)
    print(json.dumps(json.load(open("pretrain_compare.json")), indent=1))


if __name__ == "__main__":
    main()
