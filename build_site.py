"""Build site.json for the reader-facing page (index.html): the two final environments only.

    python3 build_site.py        reads data.json and harmful_agree.json in this folder

Examples are picked by a fixed random seed, never by hand, so they are not
cherry-picked: record_request training prompts (2 per kind) with replies of the
3 main models (seed 0) at RL steps 0, 50, and 300; record_request held-out
privacy prompts (2 per category of gate A) with the SFT and seed-0 RL replies
of the 5 methods and their blind labels; harmful_agree training claims (3 per
type) with bonus- and control-arm replies (seed 0) at steps 0, 100, and 300 and
the judge's stance; harmful_agree held-out claim pairs (6 harmful/harmless
pairs and 2 true claims) with SFT, bonus, and control replies and blind labels.
"""
import json
import random
from collections import defaultdict

SEED = 20261026
METHODS = ["raw", "asst", "ua", "asst_ra", "ua_ra"]
NAMES = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_ra": "asst + raw anneal", "ua_ra": "ua + raw anneal"}
d = json.load(open("data.json"))
h = json.load(open("harmful_agree.json"))
rng = random.Random(SEED)
out = {"methods": METHODS, "names": NAMES}

# record_request training prompts and replies through RL
gens = d["gens"]["record_request"]
by_kind = defaultdict(list)
for g in gens:
    by_kind[g["gts"]["kind"]].append(g)
rr_train = []
for kind in ("unsafe", "safe_private", "safe_public"):
    for g in rng.sample(by_kind[kind], 2):
        reps = {}
        for m in ("raw", "asst", "ua"):
            for step in (0, 50, 300):
                c = g["cells"].get(f"{m}|0|{step}")
                if c:
                    x = c.get("x", {})
                    reps[f"{m}|{step}"] = {"reply": c["out"], "reward": c["reward"],
                                           "leaked": bool(x.get("private_leaked")) if g["gts"]["unsafe"] else None,
                                           "refused_needed": bool(x.get("over_refusal")) if kind == "safe_private" else None}
        rr_train.append({"kind": kind, "prompt": g["prompt"], "requester": g["gts"]["requester"],
                         "requested": g["gts"]["requested"], "private_field": g["gts"].get("private_field"),
                         "replies": reps})
out["rr_train"] = rr_train

# record_request held-out privacy prompts (gate A)
pv = d["privacy_v2"]
items = pv["items"]
tag = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_ra": "asstra", "ua_ra": "uara"}
ex = defaultdict(dict)
for e in pv["examples"]:
    if e.get("set") != "A":
        continue
    ex[e["id"]][e["model"]] = e
cats = defaultdict(list)
for pid in sorted(ex):
    if all(f"base__{tag[m]}__s0__step0" in ex[pid] and f"record_request__{tag[m]}__s0__step300" in ex[pid] for m in METHODS):
        cats[pid.split("-")[1]].append(pid)
rr_held = []
for cat in sorted(cats):
    for pid in rng.sample(cats[cat], 2):
        reps = {}
        for m in METHODS:
            for which, key in (("sft", f"base__{tag[m]}__s0__step0"), ("rl", f"record_request__{tag[m]}__s0__step300")):
                e = ex[pid][key]
                reps[f"{m}|{which}"] = {"reply": e["reply"], "violated": e["violated"], "refused": e["refused"]}
        it = items[pid]
        rr_held.append({"id": pid, "category": cat, "prompt": it["prompt"], "aligned": it["aligned"],
                        "misaligned": it["misaligned"], "replies": reps})
out["rr_held"] = rr_held

# harmful_agree training claims and replies through RL
hmap = {"raw": "raw", "asst": "asst", "ua": "ua", "asst_ra": "asst_rawanneal", "ua_ra": "ua_rawanneal"}
ha_train = []
by_type = defaultdict(list)
for g in h["gens"]:
    by_type[g["type"]].append(g)
for t in ("H", "T", "C"):
    for g in rng.sample(by_type[t], 3 if t != "C" else 2):
        reps = {}
        for m in METHODS:
            for arm in ("bonus", "control"):
                for step in (0, 100, 300):
                    c = g["cells"].get(f"{hmap[m]}|{arm}|0|{step}")
                    if c:
                        reps[f"{m}|{arm}|{step}"] = {"reply": c["reply"], "stance": c.get("stance"), "reward": c.get("score")}
        ha_train.append({"type": t, "claim": g["claim"], "prompt": g["prompt"], "replies": reps})
out["ha_train"] = ha_train

# harmful_agree held-out (G6) claim pairs
pairs = defaultdict(dict)
for it in h["heldout"]:
    pairs[it["pair"]][it["type"]] = it
full = [p for p, v in pairs.items() if "H" in v and "T" in v]
ha_held = []
for p in rng.sample(sorted(full), 6):
    for t in ("H", "T"):
        ha_held.append(pairs[p][t])
withc = [p for p, v in pairs.items() if "C" in v]
for p in rng.sample(sorted(withc), 2):
    ha_held.append(pairs[p]["C"])
out["ha_held"] = [{"type": it["type"], "family": it["family"], "claim": it["claim"], "prompt": it["prompt"],
                   "replies": {f"{m}|{arm}": it["replies"][f"{hmap[m]}|{arm}"] for m in METHODS for arm in ("sft", "bonus", "control")
                               if f"{hmap[m]}|{arm}" in it["replies"]}} for it in ha_held]

json.dump(out, open("site.json", "w"), ensure_ascii=False)
print({k: len(v) for k, v in out.items() if isinstance(v, list)})
