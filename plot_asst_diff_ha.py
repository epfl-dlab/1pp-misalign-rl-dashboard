"""Each other method minus asst on harmful_agree G6, across model sizes (env_ha_asst_diff.png, ha_asst_diff.json).

    python3 plot_asst_diff_ha.py      on Clariden: reads the G6 final blind labels

For each size and method: the share of the 120 held-out harmful claims that the
bonus-arm RL runs agree with (mean of 3 RL seeds; "unclear" counts as not
agreeing). Plotted: rate(other) minus rate(asst), so a positive value means asst
agrees less. 95% interval: paired bootstrap over claims (one resample shared by
both methods), RL seeds fixed, 5,000 draws. 1.7B uses the main pretraining copy.
"""
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_env_summary import AQUA, BLUE, INK, INK2, ORANGE, grid

sys.path.insert(0, "/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl/scripts")
import harmful_agree_g6_multi as g6m  # noqa: E402

STORE = Path("/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/harmful_agree")
DIRS = {"0.5B": (["g6_wp_0p5b"], "0p5b_", ""), "1B": (["g6_wp_1b"], "1b_", ""), "1.7B": (["g6_v2", "g6_wk_base", "g6_wk_mid"], "", ""),
        "1.7B seed42 copy": (["g6_wl_raw", "g6_wl_asst", "g6_wl_ua", "g6_wm_a"], "", "_seed42")}
METHODS = ["raw", "asst", "ua", "asst_rawanneal", "ua_rawanneal"]
OTHERS = ["raw", "ua", "asst_rawanneal", "ua_rawanneal"]
NAME = {"raw": "raw", "ua": "ua", "asst_rawanneal": "asst + raw anneal", "ua_rawanneal": "ua + raw anneal"}
COL = {"raw": BLUE, "ua": AQUA, "asst_rawanneal": ORANGE, "ua_rawanneal": AQUA}
DASH = {"asst_rawanneal": (0, (4, 2)), "ua_rawanneal": (0, (4, 2))}
N_BOOT, SEED = 5000, 20261028


def main():
    items = [json.loads(l) for l in open(STORE / "g6_v2" / "prompts.jsonl")]
    H = sorted(it["id"] for it in items if it["type"] == "H")
    assert len(H) == 120
    out = {}
    for size, (dirs, pre, suf) in DIRS.items():
        lab = {}
        for d in dirs:
            lab.update(g6m.final_labels(STORE / d)[0])
        rate = {m: {i: sum(lab[(pre + m + suf, "bonus", s, i)] == "yes" for s in range(3)) / 3 for i in H} for m in METHODS}
        rng = random.Random(SEED)
        draws = {o: [] for o in OTHERS}
        for _ in range(N_BOOT):
            idx = [H[rng.randrange(len(H))] for _ in H]
            a = sum(rate["asst"][i] for i in idx) / len(idx)
            for o in OTHERS:
                draws[o].append(sum(rate[o][i] for i in idx) / len(idx) - a)
        out[size] = {}
        for o in OTHERS:
            d = sorted(draws[o])
            est = sum(rate[o][i] - rate["asst"][i] for i in H) / len(H)
            out[size][o] = {"diff": round(est, 4), "ci95": [round(d[int(0.025 * N_BOOT)], 4), round(d[int(0.975 * N_BOOT) - 1], 4)]}
    json.dump(out, open("ha_asst_diff.json", "w"), indent=1)
    sizes = ["0.5B", "1B", "1.7B"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    off = {"raw": -0.06, "ua": -0.02, "asst_rawanneal": 0.02, "ua_rawanneal": 0.06}
    for o in OTHERS:
        xs = [i + off[o] for i in range(3)]
        ys = [out[s][o]["diff"] for s in sizes]
        lo = [out[s][o]["diff"] - out[s][o]["ci95"][0] for s in sizes]
        hi = [out[s][o]["ci95"][1] - out[s][o]["diff"] for s in sizes]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=COL[o], lw=2, ls=DASH.get(o, "-"), marker="o", ms=7, capsize=3,
                    elinewidth=1, label=f"{NAME[o]} minus asst", zorder=3)
        c = out["1.7B seed42 copy"][o]
        ax.errorbar([2.22 + off[o]], [c["diff"]], yerr=[[c["diff"] - c["ci95"][0]], [c["ci95"][1] - c["diff"]]], color=COL[o],
                    marker="o", ms=7, mfc="#fcfcfb", capsize=3, elinewidth=1, ls="none", zorder=3)
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    ax.set_xticks([0, 1, 2, 2.22]); ax.set_xticklabels(["0.5B", "1B", "1.7B", "\n1.7B\nseed42"], fontsize=9); ax.set_xlim(-0.3, 2.45); grid(ax)
    ax.set_ylabel("Agree rate after RL, other minus asst")
    ax.set_title("harmful_agree: how much less asst agrees with dangerous claims after RL", loc="left", fontsize=11.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.text(0.01, 0.01, "Share of the 120 held-out harmful claims the model agrees with after bonus RL (mean of 3 seeds). Above 0: asst agrees less. "
             "Bars: 95% interval over\nclaims, seeds fixed. One pretraining copy at 0.5B and 1B. At 1.7B, filled = main copy, "
             "\nopen = seed42 copy (it differs from the main copy by more than the seed).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig("env_ha_asst_diff.png", dpi=170)
    for s in DIRS:
        print(s, {o: (v["diff"], v["ci95"]) for o, v in out[s].items()})


if __name__ == "__main__":
    main()
