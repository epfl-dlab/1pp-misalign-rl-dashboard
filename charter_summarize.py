#!/usr/bin/env python3
"""Summarize charter_mcq (MR-Eval, swap-debias-v1) on the 1PP SFT and RL checkpoints.

    python3 summarize.py [--dir DIR] [--out FILE]

DIR (default: this file's directory) must hold outputs/ with the 90
charter_mcq result files; FILE defaults to DIR/summary.json. The dashboard
copy (charter_summarize.py, charter_summary.json) was made with
--dir /capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/charter_mcq.

Per pretraining run: SFT accuracy, then per RL group (harmful_agree bonus,
harmful_agree control, record_request) the mean over its RL seeds and the
change from SFT. Per method: equal-weight mean over its pretraining runs.
Paired change from SFT per group: 95% bootstrap over the 678 items (one
resample shared by SFT and RL rows), seeds fixed; the same for bonus minus
control within a pretraining run. Overall and hard band.
"""
import argparse
import json
import random
from pathlib import Path

D = Path(__file__).resolve().parent
RUNS = ["raw", "raw_seed42", "asst", "asst_seed42", "ua", "ua_seed42", "asst_rawanneal",
        "asst_rawanneal_seed42", "ua_rawanneal", "ua_rawanneal_seed42"]
METHODS = {"raw": ["raw", "raw_seed42"], "asst": ["asst", "asst_seed42"], "ua": ["ua", "ua_seed42"],
           "asst_ra": ["asst_rawanneal", "asst_rawanneal_seed42"], "ua_ra": ["ua_rawanneal", "ua_rawanneal_seed42"]}
GROUPS = {"ha_bonus": "ha_bonus__{}__s{}", "ha_control": "ha_control__{}__s{}", "rr": "rr__{}__s{}"}
REPS = 5000


def load(name):
    f = D / "outputs" / f"charter_mcq__{name}__swap-debias-v1__greedy.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    rows = {r["id"]: r["samples"][0] for r in d["results"]}
    return {i: (s["score"], s["raw"]["band"]) for i, s in rows.items()}


def main():
    global D
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(D))
    ap.add_argument("--out")
    a = ap.parse_args()
    D = Path(a.dir)
    sft = {b: load(f"sft__{b}") for b in RUNS}
    assert all(sft.values()), [b for b, v in sft.items() if not v]
    items = sorted(next(iter(sft.values())))
    assert len(items) == 678
    hard = [i for i in items if sft["raw"][i][1] == "hard"]
    rl = {g: {b: [x for s in range(3) if (x := load(p.format(b, s)))] for b in RUNS} for g, p in GROUPS.items()}
    acc = lambda r, ids: sum(r[i][0] for i in ids) / len(ids)  # noqa: E731
    out = {"n_items": len(items), "n_hard": len(hard), "runs": {}, "methods": {}}
    rng = random.Random(20261022)
    boots = [[items[rng.randrange(len(items))] for _ in items] for _ in range(REPS)]
    for b in RUNS:
        o = {"sft": round(acc(sft[b], items), 4), "sft_hard": round(acc(sft[b], hard), 4)}
        for g in GROUPS:
            xs = rl[g][b]
            if not xs:
                continue
            o[g] = {"n_seeds": len(xs), "acc": round(sum(acc(x, items) for x in xs) / len(xs), 4),
                    "acc_hard": round(sum(acc(x, hard) for x in xs) / len(xs), 4),
                    "per_seed": [round(acc(x, items), 4) for x in xs]}
            o[g]["delta"] = round(o[g]["acc"] - o["sft"], 4)
            ds = sorted(sum(acc(x, bi) for x in xs) / len(xs) - acc(sft[b], bi) for bi in boots)
            o[g]["delta_ci95_items"] = [round(ds[int(0.025 * REPS)], 4), round(ds[int(0.975 * REPS) - 1], 4)]
        if "ha_control" in o:  # bonus minus control in the same pretraining run, same item resample for both arms
            xb, xc = rl["ha_bonus"][b], rl["ha_control"][b]
            f = lambda ids: sum(acc(x, ids) for x in xb) / len(xb) - sum(acc(x, ids) for x in xc) / len(xc)  # noqa: E731
            ds = sorted(f(bi) for bi in boots)
            o["bonus_minus_control"] = {"diff": round(f(items), 4), "ci95_items": [round(ds[int(0.025 * REPS)], 4),
                                        round(ds[int(0.975 * REPS) - 1], 4)], "seeds_bonus": len(xb), "seeds_control": len(xc)}
        out["runs"][b] = o
    for m, bs in METHODS.items():
        mo = {"sft": round(sum(out["runs"][b]["sft"] for b in bs) / len(bs), 4)}
        for g in GROUPS:
            have = [b for b in bs if g in out["runs"][b]]
            if have:
                mo[g] = round(sum(out["runs"][b][g]["acc"] for b in have) / len(have), 4)
                mo[f"{g}_delta"] = round(sum(out["runs"][b][g]["delta"] for b in have) / len(have), 4)
                mo[f"{g}_runs"] = len(have)
        out["methods"][m] = mo
    Path(a.out or D / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{'run':24} {'SFT':>6} {'HA bonus':>9} {'d':>6} {'HA ctrl':>8} {'d':>6} {'RR':>6} {'d':>6}")
    for b, o in out["runs"].items():
        cell = lambda g: (f"{o[g]['acc']:9.3f} {o[g]['delta']:+6.3f}" if g in o else f"{'-':>9} {'':>6}")  # noqa: E731
        print(f"{b:24} {o['sft']:6.3f} {cell('ha_bonus')} {cell('ha_control')} {cell('rr')}")
    print()
    for m, o in out["methods"].items():
        print(m, o)


if __name__ == "__main__":
    main()
