"""record_request held-out privacy test across model sizes (env_record_request_sizes.png).

    python3 plot_sizes_rr.py      reads privacy_wp_0p5b.json, privacy_wp_1b.json, and cross_env.json in this folder

One outcome per reply (leaked, refused without leaking, other), before RL (SFT)
and after RL (mean of 3 RL seeds), for the five methods at 0.5B, 1B, and 1.7B
(main pretraining copy). 180 held-out requests, blind labels.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_env_summary import AQUA, BLUE, GRAY, INK, INK2, ORANGE, grid

METHODS = ["raw", "asst", "ua", "asst_rawanneal", "ua_rawanneal"]
LABEL = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asst + RA", "ua_rawanneal": "ua + RA"}
COL = {"raw": BLUE, "asst": ORANGE, "ua": AQUA, "asst_rawanneal": ORANGE, "ua_rawanneal": AQUA}
TAG = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asstra", "ua_rawanneal": "uara"}
CE = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_rawanneal": "asst_ra", "ua_rawanneal": "ua_ra"}


def outcomes():
    out = {}
    for size, f, pre in (("0.5B", "privacy_wp_0p5b.json", "p5"), ("1B", "privacy_wp_1b.json", "b1")):
        o = json.load(open(f))["outcome"]
        out[size] = {m: {st: (o[pre + TAG[m]][st]["leaked"], o[pre + TAG[m]][st]["refused"]) for st in ("sft", "rl")} for m in METHODS}
    ce = json.load(open("cross_env.json"))["out"]["runs"]
    out["1.7B"] = {m: {"sft": (ce[CE[m]]["rr_sft"], ce[CE[m]]["rr_cref_sft"]), "rl": (ce[CE[m]]["rr_rl"], ce[CE[m]]["rr_cref_rl"])}
                   for m in METHODS}
    return out


def main():
    data = outcomes()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharey=True)
    w = 0.38
    for ax, size in zip(axes, ("0.5B", "1B", "1.7B")):
        for ci, m in enumerate(METHODS):
            for stage, xo in (("sft", -w / 2 - 0.02), ("rl", w / 2 + 0.02)):
                leak, ref = data[size][m][stage]
                x = ci * 1.2 + xo
                ax.bar(x, leak, w, color=COL[m] if stage == "rl" else GRAY, alpha=0.6 if stage == "rl" and "rawanneal" in m else 1.0, zorder=2)
                ax.bar(x, ref, w, bottom=leak, color="#d9d7cf", zorder=2, edgecolor="#fcfcfb", lw=1)
                ax.bar(x, 1 - leak - ref, w, bottom=leak + ref, color="#fcfcfb", edgecolor="#c3c2b7", lw=0.6, zorder=2)
                ax.text(x, leak + 0.015, f"{leak:.2f}", ha="center", va="bottom", fontsize=8, color=INK,
                        fontweight="bold" if stage == "rl" else None)
                ax.text(x, -0.03, "SFT" if stage == "sft" else "RL", ha="center", va="top", fontsize=7.5, color=INK2)
        ax.set_xticks([ci * 1.2 for ci in range(5)]); ax.set_xticklabels([LABEL[m] for m in METHODS])
        ax.tick_params(axis="x", pad=15, length=0); ax.set_ylim(0, 1.02); grid(ax)
        ax.set_title(size + (" (main copy)" if size == "1.7B" else ""), loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel("Share of held-out replies")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="leaked (colored by method; gray = SFT)"), Patch(color="#d9d7cf", label="refused, no leak"),
                        Patch(facecolor="#fcfcfb", edgecolor="#c3c2b7", label="other (evasive or garbled)")],
               frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.06))
    fig.suptitle("record_request across model sizes: held-out privacy requests before and after RL",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "One outcome per reply; RL bars = mean of 3 RL seeds; 180 held-out requests, blind AI labels; RA = raw anneal. "
             "One pretraining copy per size; the 0.5B and 1B batches have one label per reply.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94)); fig.savefig("env_record_request_sizes.png", dpi=170)


if __name__ == "__main__":
    main()
