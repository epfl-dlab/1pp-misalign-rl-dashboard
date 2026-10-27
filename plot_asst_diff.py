"""Each other method minus asst on record_request, across model sizes (env_rr_asst_diff.png, rr_asst_diff.json).

    python3 plot_asst_diff.py      on Clariden: reads the privacy v2 blind labels and keys

For each size and method: the leak (violated) rate after RL (mean of 3 RL
seeds) on the 180 held-out gate A requests. Plotted: rate(other) minus
rate(asst), so a positive value means asst leaks less. 95% interval: paired bootstrap over requests (one
resample shared by both methods), RL seeds fixed, 5,000 draws. 1.7B uses the
main pretraining copy; its labels come from the same batches as cross_env.py.
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_env_summary import AQUA, BLUE, INK, INK2, ORANGE, grid

PV = Path("/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/prev_envs/privacy_v2")
BATCHES = [("primary_key.jsonl", "primary_labels.jsonl", "primary"), ("extA_key.jsonl", "extA_labels.jsonl", "extA"),
           ("rawanneal_key.jsonl", "rawanneal_labels.jsonl", "rawanneal"), ("wp_0p5b_key.jsonl", "wp_0p5b_labels.jsonl", "wp_0p5b"),
           ("wp_1b_key.jsonl", "wp_1b_labels.jsonl", "wp_1b")]
TAG = {"0.5B": {"raw": "p5raw", "asst": "p5asst", "ua": "p5ua", "asst_rawanneal": "p5asstra", "ua_rawanneal": "p5uara"},
       "1B": {"raw": "b1raw", "asst": "b1asst", "ua": "b1ua", "asst_rawanneal": "b1asstra", "ua_rawanneal": "b1uara"},
       "1.7B": {"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asstra", "ua_rawanneal": "uara"}}
OTHERS = ["raw", "ua", "asst_rawanneal", "ua_rawanneal"]
NAME = {"raw": "raw", "ua": "ua", "asst_rawanneal": "asst + raw anneal", "ua_rawanneal": "ua + raw anneal"}
COL = {"raw": BLUE, "ua": AQUA, "asst_rawanneal": ORANGE, "ua_rawanneal": AQUA}
DASH = {"asst_rawanneal": (0, (4, 2)), "ua_rawanneal": (0, (4, 2))}
N_BOOT, SEED = 5000, 20261027


def labels():
    lab = {}
    for keyf, labf, split in BATCHES:
        L = {r["id"]: r for r in map(json.loads, open(PV / "labels" / labf))}
        for k in map(json.loads, open(PV / "labels_key" / keyf)):
            if k["split"] == split and k["set"] == "A":
                lab[(k["model"], k["id"])] = float(L[k["blind_id"]]["violated"] == "yes")
    return lab


def main():
    lab = labels()
    out = {}
    for size, tags in TAG.items():
        prompts = sorted({p for (m, p) in lab if m == f"base__{tags['asst']}__s0__step0"})
        assert len(prompts) == 180, (size, len(prompts))

        def per_prompt(m):  # leak rate per request after RL: mean over the 3 RL seeds
            t = tags[m]
            return {p: sum(lab[(f"record_request__{t}__s{k}__step300", p)] for k in range(3)) / 3 for p in prompts}
        rise = {m: per_prompt(m) for m in tags}
        rng = random.Random(SEED)
        draws = {o: [] for o in OTHERS}
        for _ in range(N_BOOT):
            ps = [prompts[rng.randrange(len(prompts))] for _ in prompts]
            a = sum(rise["asst"][p] for p in ps) / len(ps)
            for o in OTHERS:
                draws[o].append(sum(rise[o][p] for p in ps) / len(ps) - a)
        out[size] = {}
        for o in OTHERS:
            d = sorted(draws[o])
            est = sum(rise[o][p] - rise["asst"][p] for p in prompts) / len(prompts)
            out[size][o] = {"diff": round(est, 4), "ci95": [round(d[int(0.025 * N_BOOT)], 4), round(d[int(0.975 * N_BOOT) - 1], 4)]}
    json.dump(out, open("rr_asst_diff.json", "w"), indent=1)
    sizes = list(TAG)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    off = {"raw": -0.06, "ua": -0.02, "asst_rawanneal": 0.02, "ua_rawanneal": 0.06}
    for o in OTHERS:
        xs = [i + off[o] for i in range(3)]
        ys = [out[s][o]["diff"] for s in sizes]
        lo = [out[s][o]["diff"] - out[s][o]["ci95"][0] for s in sizes]
        hi = [out[s][o]["ci95"][1] - out[s][o]["diff"] for s in sizes]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=COL[o], lw=2, ls=DASH.get(o, "-"), marker="o", ms=7, capsize=3,
                    elinewidth=1, label=f"{NAME[o]} minus asst", zorder=3)
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(sizes); ax.set_xlim(-0.3, 2.3); grid(ax)
    ax.set_ylabel("Leak rate after RL, other minus asst")
    ax.set_title("record_request: how much less asst leaks after RL than each other method", loc="left", fontsize=11.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.text(0.01, 0.01, "Share of the 180 held-out requests where the reply leaked, after RL (mean of 3 seeds). Above 0: asst leaks less. "
             "Bars: 95% interval over requests,\nseeds fixed. One pretraining copy per size (1.7B: main copy). "
             "1.7B labels come from separate batches per method.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("env_rr_asst_diff.png", dpi=170)
    for s in sizes:
        print(s, {o: (v["diff"], v["ci95"]) for o, v in out[s].items()})


if __name__ == "__main__":
    main()
