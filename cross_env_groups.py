"""Keyword groups for the Patterns section (descriptive only). Usage: python3 cross_env_groups.py cross_env.json"""
import json, re, sys, numpy as np
d = json.load(open(sys.argv[1]))
items = {json.loads(x)["id"]: json.loads(x) for x in open("/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store/harmful_agree/g6_v2/prompts.jsonl")}
pit = json.load(open("/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl-dashboard/data.json"))["privacy_v2"]["items"]
runs = list(next(iter(d["out"]["ha_items"].values())).keys())
ha, rr = d["out"]["ha_items"], d["out"]["rr_items"]
can = lambda i: items[i]["claim"].lower().startswith("you can")
mal = re.compile(r"get even|revenge|settle the score|stabb|screwed me|so (he|she|they) gets? fired|sabotag|pay (him|her) back|teach (him|her)|humiliat|ruin|expose (him|her)|get back at", re.I)
hostile = lambda p: bool(mal.search(pit[p]["prompt"]))
task = re.compile(r"\b(draft|write|compose|email|text message|letter|message)\b", re.I)
writing = lambda p: bool(task.search(pit[p]["prompt"]))
res = {"runs": runs, "groups": {}}
def add(name, src, pred, k):
    get = (lambda v, r: v[r]) if k is None else (lambda v, r: v[r][k])
    a = [i for i in src if pred(i)]; b = [i for i in src if not pred(i)]
    res["groups"][name] = {"n_in": len(a), "n_out": len(b),
        "in": {r: round(float(np.mean([get(src[i], r) for i in a])), 3) for r in runs},
        "out": {r: round(float(np.mean([get(src[i], r) for i in b])), 3) for r in runs}}
add("ha_you_can", ha, can, None)
add("rr_hostile_violated", rr, hostile, 0)
add("rr_hostile_refused", rr, hostile, 1)
add("rr_writing_violated", rr, writing, 0)
for g, v in res["groups"].items():
    print(g, v["n_in"], v["n_out"], "mean in", round(np.mean(list(v["in"].values())), 2), "out", round(np.mean(list(v["out"].values())), 2))
    print("   in ", v["in"]); print("   out", v["out"])
json.dump(res, open(sys.argv[2], "w"), indent=1)
