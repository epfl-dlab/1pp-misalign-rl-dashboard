import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, "/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl/scripts")
import harmful_agree_g6_multi as g
HA = "/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/harmful_agree"
PV = Path("/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/prev_envs/privacy_v2")
RUNS = [("raw","raw","raw"),("raw_s42","raw_seed42","raws42"),("asst","asst","asst"),("asst_s42","asst_seed42","assts42"),
        ("ua","ua","ua"),("ua_s42","ua_seed42","uas42"),("asst_ra","asst_rawanneal","asstra"),("asst_ra_s42","asst_rawanneal_seed42","asstras42"),
        ("ua_ra","ua_rawanneal","uara"),("ua_ra_s42","ua_rawanneal_seed42","uaras42")]
# harmful_agree labels
lab = {}
for d in ("g6_v2","g6_wk_base","g6_wk_mid","g6_wl_raw","g6_wl_asst","g6_wl_ua","g6_wm_a"):
    lab.update(g.final_labels(Path(f"{HA}/{d}"))[0])
items = {json.loads(x)["id"]: json.loads(x) for x in open(f"{HA}/g6_v2/prompts.jsonl")}
H = sorted(i for i, it in items.items() if it["type"] == "H")
# record_request labels (violated, refused)
rl = {}
for keyf, labf, split in [("primary_key.jsonl","primary_labels.jsonl","primary"),("extA_key.jsonl","extA_labels.jsonl","extA"),
                          ("rawanneal_key.jsonl","rawanneal_labels.jsonl","rawanneal"),("seed42_key.jsonl","seed42_labels.jsonl","seed42"),
                          ("ras42_key.jsonl","ras42_labels.jsonl","ras42")]:
    labels = {r["id"]: r for r in map(json.loads, open(PV/"labels"/labf))}
    for k in map(json.loads, open(PV/"labels_key"/keyf)):
        if k["split"] == split and k["set"] == "A":
            rl[(k["model"], k["id"])] = labels[k["blind_id"]]
P = sorted({p for (m,p) in rl if m == "base__raws42__s0__step0"})
pitems = json.load(open("/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl-dashboard/data.json"))["privacy_v2"]["items"]
out = {"runs": {}, "ha_items": {}, "rr_items": {}}
for name, hb, rb in RUNS:
    ha_sft = np.array([lab[(hb,"sft",None,i)]=="yes" for i in H], float)
    ha_rl = np.mean([[lab[(hb,"bonus",s,i)]=="yes" for i in H] for s in range(3)], 0)
    rr_sft = np.array([rl[(f"base__{rb}__s0__step0",p)]["violated"]=="yes" for p in P], float)
    rr_rl = np.mean([[rl[(f"record_request__{rb}__s{s}__step300",p)]["violated"]=="yes" for p in P] for s in range(3)], 0)
    rr_ref_sft = np.array([rl[(f"base__{rb}__s0__step0",p)]["refused"]=="yes" for p in P], float)
    rr_ref_rl = np.mean([[rl[(f"record_request__{rb}__s{s}__step300",p)]["refused"]=="yes" for p in P] for s in range(3)], 0)
    # single outcome label: leaked (violated), refused (refused and not violated), other
    clean = lambda r: r["refused"] == "yes" and r["violated"] != "yes"  # noqa: E731
    rr_cref_sft = np.array([clean(rl[(f"base__{rb}__s0__step0",p)]) for p in P], float)
    rr_cref_rl = np.mean([[clean(rl[(f"record_request__{rb}__s{s}__step300",p)]) for p in P] for s in range(3)], 0)
    out["runs"][name] = dict(rr_cref_sft=rr_cref_sft.mean(), rr_cref_rl=rr_cref_rl.mean(), ha_sft=ha_sft.mean(), ha_rl=ha_rl.mean(), rr_sft=rr_sft.mean(), rr_rl=rr_rl.mean(), rr_ref_sft=rr_ref_sft.mean(), rr_ref_rl=rr_ref_rl.mean())
    for i, v in zip(H, ha_rl): out["ha_items"].setdefault(i, {})[name] = float(v)
    for p, v, rf, cf in zip(P, rr_rl, rr_ref_rl, rr_cref_rl): out["rr_items"].setdefault(p, {})[name] = (float(v), float(rf), float(cf))
print(f"{'run':12} {'HA sft':>6} {'HA RL':>6} {'RR sft':>6} {'RR RL':>6} {'RR refuse sft->RL':>18}")
for n, r in out["runs"].items():
    print(f"{n:12} {r['ha_sft']:6.2f} {r['ha_rl']:6.2f} {r['rr_sft']:6.2f} {r['rr_rl']:6.2f} {r['rr_ref_sft']:8.2f} -> {r['rr_ref_rl']:.2f}"
          f"   refused, no leak: {r['rr_cref_sft']:.2f} -> {r['rr_cref_rl']:.2f}   other: {1 - r['rr_sft'] - r['rr_cref_sft']:.2f} -> {1 - r['rr_rl'] - r['rr_cref_rl']:.2f}")
ha = [out["runs"][n]["ha_rl"] for n,_,_ in RUNS]; rr = [out["runs"][n]["rr_rl"] for n,_,_ in RUNS]
print("corr across 10 runs (HA RL vs RR RL):", round(float(np.corrcoef(ha, rr)[0,1]),2))
# by category / family
def bycat(get_items, key, getter):
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for it, vals in get_items.items():
        c = key(it)
        for n, v in vals.items(): res[c][n].append(getter(v))
    return res
fam = bycat(out["ha_items"], lambda i: items[i]["family"], lambda v: v)
print("\nharmful_agree after RL by topic family (mean over items):")
names = [n for n,_,_ in RUNS]
print(f"{'family':26}", " ".join(f"{n[:10]:>10}" for n in names))
for f, d in sorted(fam.items()): print(f"{f:26}", " ".join(f"{np.mean(d[n]):10.2f}" for n in names))
cat = bycat(out["rr_items"], lambda p: p.split("-")[1], lambda v: v[0])
print("\nrecord_request after RL by request category (violated):")
for f, d in sorted(cat.items()): print(f"{f:26}", " ".join(f"{np.mean(d[n]):10.2f}" for n in names))
catr = bycat(out["rr_items"], lambda p: p.split("-")[1], lambda v: v[1])
print("\nrecord_request after RL by category (refused):")
for f, d in sorted(catr.items()): print(f"{f:26}", " ".join(f"{np.mean(d[n]):10.2f}" for n in names))
# items robust (all runs resist) and items all fail
hr = {i: np.mean(list(v.values())) for i, v in out["ha_items"].items()}
rr_ = {p: np.mean([x[0] for x in v.values()]) for p, v in out["rr_items"].items()}
print("\nHA items: mean agree after RL across 10 runs: share of items <0.2:", round(np.mean([x<0.2 for x in hr.values()]),2), " >0.9:", round(np.mean([x>0.9 for x in hr.values()]),2))
print("RR items: mean violated after RL: share <0.1:", round(np.mean([x<0.1 for x in rr_.values()]),2), " >0.8:", round(np.mean([x>0.8 for x in rr_.values()]),2))
json.dump({"out": out, "hr": hr, "rr": rr_}, open(sys.argv[1], "w"), default=float)
