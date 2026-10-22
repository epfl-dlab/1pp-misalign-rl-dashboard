"""charter_mcq before and after RL, per pretraining method (env_charter.png, charter.json).

    python3 plot_charter.py      on Clariden; reads misalign-rl-store/charter_mcq/summary.json

charter_mcq is MR-Eval's value-charter MCQ (678 items, swap-debiased
first-token logprob, chance 0.25). Left: accuracy of the SFT model and after
each kind of RL (step 300). Bar = equal-weight mean of the method's pretraining
runs, each first averaged over its RL seeds; dots = pretraining runs (filled =
main, hollow = seed42). Right: change from SFT, same averaging. The
harmful_agree control bars have fewer runs: no control runs exist for
ua_seed42 and the two raw-anneal seed42 models, and asst_seed42 has 2 seeds.
Descriptive: no intervals; per-run item-bootstrap intervals are in the store's
summary.json.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/charter_mcq/summary.json"))
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
GROUPS = [("sft", "SFT (before RL)", "#a3a29b"), ("ha_bonus", "harmful_agree, bonus arm", "#e34948"),
          ("ha_control", "harmful_agree, control arm", "#2a78d6"), ("rr", "record_request", "#8a5cc4")]
METHODS = [("raw", "raw", ["raw", "raw_seed42"]), ("asst", "asst", ["asst", "asst_seed42"]), ("ua", "ua", ["ua", "ua_seed42"]),
           ("asst_ra", "asst +\nraw anneal", ["asst_rawanneal", "asst_rawanneal_seed42"]),
           ("ua_ra", "ua +\nraw anneal", ["ua_rawanneal", "ua_rawanneal_seed42"])]
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def value(run, g, delta):
    o = S["runs"][run]
    if g == "sft":
        return None if delta else o["sft"]
    if g not in o:
        return None
    return o[g]["delta"] if delta else o[g]["acc"]


def panel(ax, groups, delta):
    w = 0.8 / len(groups); out = {}
    for i, (m, _, runs) in enumerate(METHODS):
        for j, (g, _, col) in enumerate(groups):
            vals = [(r, value(r, g, delta)) for r in runs]
            vals = [(r, v) for r, v in vals if v is not None]
            if not vals:
                continue
            x = i + (j - (len(groups) - 1) / 2) * w
            mean = sum(v for _, v in vals) / len(vals)
            out[f"{m}|{g}"] = {"mean": round(mean, 4), "runs": {r: v for r, v in vals}}
            partial = len(vals) < len(runs)  # fewer runs than the method has: not comparable to the full bars
            ax.bar(x, mean, w * 0.92, color=col, alpha=0.35 if partial else 1.0, hatch="///" if partial else None,
                   edgecolor=col, lw=0.8 if partial else 0, zorder=2)
            for r, v in vals:
                seed42 = r.endswith("seed42")
                ax.scatter([x], [v], s=18, zorder=4, facecolor=SURF if seed42 else INK2, edgecolor=INK2, lw=0.9)
            if delta:  # label beyond the bar end and every dot, so nothing overlaps
                end = max([mean] + [v for _, v in vals]) if mean >= 0 else min([mean] + [v for _, v in vals])
                ax.text(x, end + (0.006 if mean >= 0 else -0.006), f"{mean:+.2f}", ha="center",
                        va="bottom" if mean >= 0 else "top", fontsize=8, color=INK)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([n for _, n, _ in METHODS], fontsize=9.5)
    ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    return out


def main():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5.0), gridspec_kw={"width_ratios": [1.15, 1, 0.6]})
    acc = panel(a1, GROUPS, False)
    a1.axhline(0.25, color=INK2, lw=0.8, ls=":"); a1.text(-0.45, 0.255, "chance", fontsize=8, color=INK2, ha="left", va="bottom")
    a1.set_ylim(0, 0.75); a1.set_ylabel("charter_mcq accuracy (678 items)")
    a1.set_title("Accuracy before and after RL", loc="left", fontsize=11, color=INK)
    d = panel(a2, GROUPS[1:], True)
    a2.axhline(0, color=INK2, lw=0.9); a2.set_ylim(-0.12, 0.12); a2.set_ylabel("Change from SFT")
    a2.set_title("Change from SFT", loc="left", fontsize=11, color=INK)
    # matched contrast: bonus minus control within each pretraining run that has control runs
    mc = {}
    for i, (m, n, runs) in enumerate(METHODS):
        have = [r for r in runs if "ha_control" in S["runs"][r]]
        diffs = [S["runs"][r]["ha_bonus"]["acc"] - S["runs"][r]["ha_control"]["acc"] for r in have]
        mean = sum(diffs) / len(diffs)
        mc[m] = {"mean": round(mean, 4), "runs": {r: round(x, 4) for r, x in zip(have, diffs)}}
        a3.bar(i, mean, 0.6, color="#7a2e2e", zorder=2)
        for r, x in zip(have, diffs):
            a3.scatter([i], [x], s=18, zorder=4, facecolor=SURF if r.endswith("seed42") else INK2, edgecolor=INK2, lw=0.9)
        lo = min([mean] + diffs)
        a3.text(i, lo - 0.006, f"{mean:+.2f}\n({len(have)} run{'s' if len(have) > 1 else ''})", ha="center", va="top", fontsize=8, color=INK)
    a3.axhline(0, color=INK2, lw=0.9); a3.set_ylim(-0.16, 0.02); a3.set_xticks(range(len(METHODS)))
    a3.set_xticklabels(["raw", "asst", "ua", "asst\n+ RA", "ua\n+ RA"], fontsize=9); a3.grid(axis="y", color="#e8e7e2", lw=0.8); a3.set_axisbelow(True)
    a3.set_title("Bonus minus control, matched runs", loc="left", fontsize=11, color=INK)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in GROUPS]
              + [Patch(facecolor="#9a9a94", alpha=0.35, hatch="///", edgecolor="#9a9a94", label="fewer runs than the method (main run only)"),
                 Line2D([], [], marker="o", ls="", color=INK2, ms=5, label="main pretraining run"),
                 Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, ms=5, label="seed42 pretraining run")],
              frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=7)
    fig.suptitle("Charter alignment (charter_mcq) before and after RL, per pretraining method", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Bar = mean of the method's pretraining runs (main and seed42), each averaged over its RL seeds; dots = runs. RL checkpoints at step 300.\n"
             "No harmful_agree control runs for ua seed42 and the two raw-anneal seed42 models (hatched bars use the main run only; compare them only in the right panel,\n"
             "which pairs bonus and control within each run). asst seed42 control has 2 seeds. Forced-choice test: it measures picking the charter-aligned option, not\n"
             "free-text behavior. Descriptive; per-run item-bootstrap intervals are in charter_mcq/summary.json.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.12, 1, 0.89)); fig.savefig("env_charter.png", dpi=170)
    json.dump({"accuracy": acc, "change": d, "bonus_minus_control_matched": mc}, open("charter.json", "w"), indent=1)


if __name__ == "__main__":
    main()
