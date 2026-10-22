"""One figure: harmful behavior before and after RL in both environments, pooled over models and seeds.

    python3 plot_overall.py      on Clariden; writes env_overall.png

Unit = one RL run (model x RL seed); its "before" is its model's SFT rate on the
same held-out prompts. Bars = mean over runs. Descriptive 95% intervals: 10,000 bootstrap draws
that resample models, then RL seeds within each drawn model (runs of one
model are not independent). harmful_agree: 8 models x 3 bonus seeds, share
of 120 held-out harmful claims the reply agrees with. record_request: 8
models x 3 seeds, share of 180 held-out privacy requests where the reply
helps. No control arm is subtracted.
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

S = "/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/harmful_agree"
GRAY, RED, INK, INK2, SURF = "#a3a29b", "#e34948", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def harmful_agree():
    lab = {}
    for d in ("g6_v2", "g6_wk_base", "g6_wk_mid", "g6_wl_raw", "g6_wl_asst", "g6_wl_ua", "g6_wm_a"):
        l = g.final_labels(Path(f"{S}/{d}"))[0]
        if set(l) & set(lab):
            raise SystemExit(f"{d}: labels overlap another dir")
        lab.update(l)
    items = {json.loads(x)["id"]: json.loads(x) for x in open(f"{S}/g6_v2/prompts.jsonl")}
    H = [i for i, it in items.items() if it["type"] == "H"]
    rate = lambda m, arm, sd: float(np.mean([lab[(m, arm, sd, i)] == "yes" for i in H]))  # noqa: E731
    models = ["raw", "asst", "ua", "asst_rawanneal", "ua_rawanneal", "raw_seed42", "asst_seed42", "ua_seed42",
              "asst_rawanneal_seed42", "ua_rawanneal_seed42"]
    return {m: (rate(m, "sft", None), [rate(m, "bonus", s) for s in (0, 1, 2)]) for m in models}


def record_request():
    v = dict(json.load(open("data.json"))["privacy_v2"]["rawanneal"]["violated"])
    v.update(json.load(open("privacy_seed42.json"))["violated"])  # seed42 batch
    v.update(json.load(open("privacy_ras42.json"))["violated"])  # ras42 batch
    return {m: (v[m]["sft"], v[m]["rl_seeds"]) for m in ["raw", "asst", "ua", "asstra", "uara", "raws42", "assts42", "uas42", "asstras42", "uaras42"]}


def stats(data, reps=10000):
    """Means over runs, and descriptive bootstrap 95% intervals (models shown, then seeds within a model)."""
    def summary(sample):
        pre = [data[m][0] for m, sds in sample for _ in sds]
        post = [x for m, sds in sample for x in sds]
        return np.mean(pre), np.mean(post), np.mean(post) - np.mean(pre)
    full = [(m, data[m][1]) for m in data]
    est = summary(full)
    rng = random.Random(0); ms = list(data); draws = []
    for _ in range(reps):
        sample = []
        for m in (rng.choice(ms) for _ in ms):
            sample.append((m, [rng.choice(data[m][1]) for _ in data[m][1]]))
        draws.append(summary(sample))
    draws = np.array(draws)
    ci = np.percentile(draws, [2.5, 97.5], axis=0)
    return est, ci, len(ms), sum(len(data[m][1]) for m in data)


def main():
    envs = [("harmful_agree", "Agrees with a harmful false claim", harmful_agree()),
            ("record_request", "Helps invade privacy in plain chat", record_request())]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.34; out = {}
    for i, (name, what, data) in enumerate(envs):
        est, ci, nm, nr = stats(data); out[name] = {"pre": est[0], "post": est[1], "change": est[2], "ci": ci.tolist(),
                                                     "models": nm, "runs": nr}
        for j, (val, lo, hi, col, lab) in enumerate([(est[0], ci[0][0], ci[1][0], GRAY, "Before RL (SFT)"),
                                                     (est[1], ci[0][1], ci[1][1], RED, "After RL")]):
            x = i + (j - 0.5) * (w + 0.03)
            ax.bar(x, val, w, color=col, zorder=2, label=lab if i == 0 else None)
            ax.errorbar([x], [val], yerr=[[val - lo], [hi - val]], fmt="none", ecolor=INK, elinewidth=1.4, capsize=4, zorder=3)
            ax.text(x, hi + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold" if j else "normal")
        ax.text(i, max(ci[1][1], ci[1][0]) + 0.11, f"change {est[2]:+.2f} [{ci[0][2]:+.2f}, {ci[1][2]:+.2f}]",
                ha="center", fontsize=10, color=INK)
        ax.text(i, -0.13, f"{what}\n{nm} models × 3 RL seeds = {nr} runs", ha="center", va="top", fontsize=9, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels([e[0] for e in envs], fontsize=11, color=INK); ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Share of held-out replies"); ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.suptitle("Harmful behavior on held-out prompts, before and after RL", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Mean over RL runs. Whiskers and brackets: descriptive 95% bootstrap over the models shown, then their 3 RL seeds; it leaves out prompt\n"
             "sampling noise, and the models are chosen, not sampled. Both count each of the 5 methods twice (main and seed42 runs).\n"
             "No control arm subtracted; harmful_agree's control runs agree with 0.00-0.01 of these claims. record_request: most of the rise matches less refusal.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94)); fig.savefig("env_overall.png", dpi=170)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
