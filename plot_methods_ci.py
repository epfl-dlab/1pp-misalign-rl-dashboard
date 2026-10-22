"""Post-RL harmful behavior per pretraining method, one bar each, with 95% intervals (env_methods_ci.png, methods_ci.json).

    python3 plot_methods_ci.py      on Clariden; reads the label stores

Bar = mean over the method's pretraining runs (main and seed42), each run first averaged over its 3 RL seeds, so every run counts
once. Error bar = 95% bootstrap (5,000 draws): one resample of the held-out
prompts per draw, shared by all methods, and a resample of the 3 RL seeds
within each run. It covers prompt and RL-seed noise, not pretraining-seed
noise (at most 2 runs per method); the run means are shown as dots.
methods_ci.json also gives paired 95% intervals for each difference between
two methods, from the same draws (same conditional scope).
harmful_agree: agreement with 120 held-out harmful claims, bonus arm (final
blind labels). record_request: violated on 180 gate A privacy prompts (blind
labels; main and anneal runs from the primary, extA, and raw-anneal batches,
seed42 runs from the seed42 and ras42 batches).
"""
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl/scripts")
import harmful_agree_g6_multi as g  # noqa: E402

HA = "/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/harmful_agree"
PV = Path("/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/prev_envs/privacy_v2")
BLUE, ORANGE, AQUA, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
METHODS = [("raw", "raw", BLUE, 1.0), ("asst", "asst", ORANGE, 1.0), ("ua", "ua", AQUA, 1.0),
           ("asst_ra", "asst +\nraw anneal", ORANGE, 0.55), ("ua_ra", "ua +\nraw anneal", AQUA, 0.55)]
REPS = 5000
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def harmful_agree():
    """method -> list of runs; run = (is_seed42, {seed: 0/1 array over prompts})."""
    lab = {}
    for d in ("g6_v2", "g6_wk_base", "g6_wk_mid", "g6_wl_raw", "g6_wl_asst", "g6_wl_ua", "g6_wm_a"):
        l = g.final_labels(Path(f"{HA}/{d}"))[0]
        if set(l) & set(lab):
            raise SystemExit(f"{d}: labels overlap another dir")
        lab.update(l)
    items = {json.loads(x)["id"]: json.loads(x) for x in open(f"{HA}/g6_v2/prompts.jsonl")}
    H = sorted(i for i, it in items.items() if it["type"] == "H")
    run = lambda m: {s: np.array([lab[(m, "bonus", s, i)] == "yes" for i in H], float) for s in (0, 1, 2)}  # noqa: E731
    src = {"raw": ["raw", "raw_seed42"], "asst": ["asst", "asst_seed42"], "ua": ["ua", "ua_seed42"],
           "asst_ra": ["asst_rawanneal", "asst_rawanneal_seed42"], "ua_ra": ["ua_rawanneal", "ua_rawanneal_seed42"]}
    return {k: [(m.endswith("seed42"), run(m)) for m in v] for k, v in src.items()}, len(H)


def record_request():
    lab = {}
    for keyf, labf, split in [("primary_key.jsonl", "primary_labels.jsonl", "primary"),
                              ("extA_key.jsonl", "extA_labels.jsonl", "extA"),
                              ("rawanneal_key.jsonl", "rawanneal_labels.jsonl", "rawanneal"),
                              ("seed42_key.jsonl", "seed42_labels.jsonl", "seed42"),
                              ("ras42_key.jsonl", "ras42_labels.jsonl", "ras42")]:
        labels = {r["id"]: r for r in map(json.loads, open(PV / "labels" / labf))}
        for k in map(json.loads, open(PV / "labels_key" / keyf)):
            if k["split"] == split and k["set"] == "A":
                lab[(k["model"], k["id"])] = labels[k["blind_id"]]["violated"]
    prompts = sorted({p for (m, p) in lab if m == "base__raws42__s0__step0"})
    assert len(prompts) == 180
    run = lambda m: {s: np.array([lab[(f"record_request__{m}__s{s}__step300", p)] == "yes" for p in prompts], float)  # noqa: E731
                     for s in (0, 1, 2)}
    src = {"raw": ["raw", "raws42"], "asst": ["asst", "assts42"], "ua": ["ua", "uas42"], "asst_ra": ["asstra", "asstras42"], "ua_ra": ["uara", "uaras42"]}
    return {k: [(m.endswith("s42"), run(m)) for m in v] for k, v in src.items()}, len(prompts)


def summarize(data, n):
    est = {k: float(np.mean([np.mean([v.mean() for v in sd.values()]) for _, sd in runs])) for k, runs in data.items()}
    rng = random.Random(0); draws = {k: [] for k in data}
    for _ in range(REPS):
        idx = np.array([rng.randrange(n) for _ in range(n)])
        for k, runs in data.items():
            draws[k].append(np.mean([np.mean([sd[rng.choice((0, 1, 2))][idx].mean() for _ in range(3)]) for _, sd in runs]))
    res = {k: {"mean": round(est[k], 3), "ci95": [round(float(x), 3) for x in np.percentile(draws[k], [2.5, 97.5])],
               "run_means": [[rep, round(float(np.mean([v.mean() for v in sd.values()])), 3)] for rep, sd in data[k]]}
           for k in data}
    # paired contrasts: both methods use the same prompt resample in each draw
    ks = list(data)
    res["contrasts"] = {f"{a}-{b}": {"diff": round(est[a] - est[b], 3), "ci95": [round(float(x), 3) for x in
                        np.percentile(np.array(draws[a]) - np.array(draws[b]), [2.5, 97.5])]}
                        for i, a in enumerate(ks) for b in ks[i + 1:]}
    return res


def panel(ax, res, title):
    for i, (k, name, col, alpha) in enumerate(METHODS):
        r = res[k]; m, (lo, hi) = r["mean"], r["ci95"]
        ax.bar(i, m, 0.62, color=col, alpha=alpha, zorder=2)
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=4)
        for rep, v in r["run_means"]:
            ax.scatter([i + (0.17 if rep else -0.17)], [v], s=26, zorder=5, facecolor=SURF if rep else INK2, edgecolor=INK2, lw=1)
        ax.text(i, max(hi, max(v for _, v in r["run_means"])) + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=10.5,
                color=INK, fontweight="bold")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([f"{nm}\n({len(res[k]['run_means'])} run{'s' if len(res[k]['run_means']) > 1 else ''})"
                        for k, nm, _, _ in METHODS], fontsize=9.5)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)


def main():
    ha, n_ha = harmful_agree(); rr, n_rr = record_request()
    out = {"harmful_agree": summarize(ha, n_ha), "record_request": summarize(rr, n_rr)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    panel(a1, out["harmful_agree"], "harmful_agree: agrees with a harmful false claim")
    panel(a2, out["record_request"], "record_request: helps invade privacy")
    a1.set_ylabel("Share of held-out replies, after RL")
    from matplotlib.lines import Line2D
    a2.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=6, label="main pretraining run (mean of 3 RL seeds)"),
                       Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, ms=6, label="seed42 pretraining run (mean of 3 RL seeds)")],
              frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Harmful behavior after RL, per pretraining method", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Bar = mean of the method's pretraining runs (run = one pretrained model, mean of its 3 RL seeds). Error bar = 95% bootstrap over held-out prompts and RL seeds,\n"
             "conditional on these pretrained models: it leaves out pretraining-seed noise (dots) and labeling-batch effects. harmful_agree: 120 harmful claims, bonus arm. record_request: 180 privacy prompts,\n"
             "labels from several batches (the seed42 runs in two batches of their own). No control arm subtracted.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94)); fig.savefig("env_methods_ci.png", dpi=170)
    json.dump(out, open("methods_ci.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
