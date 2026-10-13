"""Dashboard data for the two promising v1 envs (record_request, review_blurb).

    python3 -I build.py extract OUT_DIR      CPU: curves, gens, held-out, audits
    python3 -I build.py judge OUT_DIR        needs MISALIGN_JUDGE_URL: re-runs the
                                             review_blurb judges on the sampled
                                             gens to recover their rationales

Writes OUT_DIR/data.json (and OUT_DIR/rationale.json in judge mode).
"""
import asyncio
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

STORE = "/capstor/store/cscs/swissai/ab023/rsinghal/misalign-rl-store"
CK = f"{STORE}/checkpoints/1pp-misalign-rl"
HO = f"{STORE}/heldout"
REPO = "/capstor/store/cscs/swissai/ab023/rsinghal/1pp-misalign-rl"
CONDS = ["raw", "asst", "ua"]
SEEDS = [0, 1, 2]
ENVS = ["record_request", "review_blurb"]
GEN_STEPS = [0, 25, 50, 75, 100, 150, 200, 300]
N_PROMPTS = 24
EXTRA = {
    "record_request": ["unsafe", "private_leaked", "private_filled", "fields_correct", "over_refusal",
                       "valid_json", "a"],
    "review_blurb": ["a", "facts_traced", "words", "length_ok", "appeal", "h", "gate", "paid_hidden",
                     "defect_omitted", "defect_keys_present", "n_invented", "exploit_invented"],
}


def run_dir(env, cond, seed):
    suf = "_probe" if env == "review_blurb" else ""
    return f"{CK}/1pp_1p7b_{cond}_sft_{env}_added1.0_muon_s{seed}{suf}"


def load(path):
    return [json.loads(l) for l in open(path)]


def strip(text):
    t = text[5:] if text.startswith("user\n") else text
    return (t[: -len("assistant\n")] if t.endswith("assistant\n") else t).strip()


def num(v):
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return v


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def point(env, rows):
    f = lambda k: [float(r[k]) for r in rows if k in r]
    p = {"reward": mean(f("reward") or f("score")), "n": len(rows)}
    if env == "record_request":
        uns = [r for r in rows if float(r["unsafe"]) == 1.0]
        legit = [r for r in rows if float(r["unsafe"]) == 0.0]
        p["exploit"] = mean(float(r["exploit"]) for r in uns)
        p["over_refusal"] = mean(float(r["over_refusal"]) for r in legit)
        p["fields_correct"] = mean(f("fields_correct"))
        p["valid_json"] = mean(f("valid_json"))
    else:
        p["exploit"] = mean(f("exploit"))
        p["exploit_invented"] = mean(f("exploit_invented"))
        p["appeal"] = mean(f("appeal"))
        p["a"] = mean(f("a"))
        p["words"] = mean(f("words"))
        p["judge_ok"] = mean(f("judge_ok"))
    return p


def spec_text():
    out = {}
    for env in ENVS:
        src = open(f"{REPO}/rl/rewards/{env}.py").read()
        doc = src.split('"""')[1].strip() if src.startswith('"""') else ""
        out[env] = {"doc": doc}
    sys.path.insert(0, REPO)
    from rl.rewards import record_request as rr, review_blurb as rb  # noqa: E402
    out["record_request"]["prompt_template"] = getattr(rr, "PROMPT", "")
    out["review_blurb"]["prompt_template"] = rb.PROMPT
    out["review_blurb"]["judges"] = {"appeal_system": rb.APPEAL_SYSTEM, "appeal_user": rb.APPEAL_USER,
                                     "detect_system": rb.DETECT_SYSTEM, "detect_user": rb.DETECT_USER}
    return out


def extract(outdir):
    data = {"spec": spec_text(), "curves": {}, "rollout": {}, "gens": {}, "runs": {}}
    for env in ENVS:
        curves, roll, cache = [], [], {}
        for c in CONDS:
            for s in SEEDS:
                d = run_dir(env, c, s)
                info = json.load(open(f"{d}/run_info.json")) if os.path.exists(f"{d}/run_info.json") else {}
                data["runs"][f"{env}|{c}|{s}"] = {"dir": d, "job": info.get("slurm_job_id"),
                                                  "commit": info.get("repo_commit")}
                pts = []
                for st in sorted(int(os.path.basename(x)[:-6]) for x in glob.glob(f"{d}/validation/*.jsonl")):
                    rows = load(f"{d}/validation/{st}.jsonl")
                    if st in GEN_STEPS:
                        cache[(c, s, st)] = rows
                    pts.append({"step": st, **point(env, rows)})
                curves.append({"cond": c, "seed": s, "points": pts})
                rp = []
                for st in range(1, 301):
                    fp = f"{d}/rollouts/{st}.jsonl"
                    if os.path.exists(fp):
                        rows = load(fp)
                        q = point(env, rows)
                        rp.append([st, q["reward"], q["exploit"]])
                roll.append({"cond": c, "seed": s, "points": rp})
            print(env, c, flush=True)
        data["curves"][env], data["rollout"][env] = curves, roll

        base = cache[("raw", 0, 0)]
        first = {}
        for r in base:
            first.setdefault(r["input"], r)
        prompts = sorted(first, key=lambda p: hashlib.sha1(p.encode()).hexdigest())
        if env == "record_request":
            uns = [p for p in prompts if float(first[p]["unsafe"]) == 1.0]
            leg = [p for p in prompts if float(first[p]["unsafe"]) == 0.0]
            pick = uns[:16] + leg[:8]
        else:
            pick = prompts[:N_PROMPTS]
        gens = []
        for i, p in enumerate(pick):
            item = {"pid": i, "prompt": strip(p), "gts": json.loads(first[p]["gts"]), "cells": {}}
            for c in CONDS:
                for s in SEEDS:
                    for st in GEN_STEPS:
                        hit = next((r for r in cache.get((c, s, st), []) if r["input"] == p), None)
                        if hit is None:
                            continue
                        item["cells"][f"{c}|{s}|{st}"] = {
                            "out": hit["output"], "reward": num(hit.get("reward", hit.get("score"))),
                            "exploit": num(hit["exploit"]),
                            "x": {k: num(hit[k]) for k in EXTRA[env] if k in hit}}
            gens.append(item)
        data["gens"][env] = gens

    # Held-out generative sets (judge v2, one-sentence reason), sample 0 only.
    keep = {f"base__{c}__s0__step0" for c in CONDS}
    keep |= {f"{e}__{c}__s{s}__step300" for e in ENVS for c in CONDS for s in SEEDS}
    prompts = {r["id"]: r for r in load(f"{HO}/prompts.jsonl")}
    ho = []
    for r in map(json.loads, open(f"{HO}/gen_ab.graded_v2.jsonl")):
        if r["model"] in keep and int(r.get("sample", 0)) == 0:
            ho.append({k: r.get(k) for k in ("model", "id", "set", "control", "reply", "label", "reason")})
    data["heldout"] = {"prompts": prompts, "rows": ho,
                       "table": open(f"{HO}/gen_table_v2_ab.txt").read(),
                       "judge_report": open(f"{HO}/label_report_v2.txt").read()}

    # Privacy transfer with opaque-id blind labels (720 replies).
    key = load(f"{HO}/keys_v2/privacy_v2_key.jsonl")
    lab, over = {}, {}
    for f in glob.glob(f"{HO}/blind_v2/labels_privacy_v2_labeler*.jsonl"):
        lab.update({r["id"]: r["label"] for r in load(f)})
    of = f"{HO}/blind_v2/labels_privacy_v2_overlap.jsonl"
    if os.path.exists(of):
        over = {r["id"]: r["label"] for r in load(of)}
    data["privacy_blind"] = [{"model": r["model"], "cell": r.get("cell"), "seed": r.get("seed"), "id": r["id"],
                              "reply": r["reply"], "label": lab.get(r["blind_id"]),
                              "overlap_label": over.get(r["blind_id"])} for r in key]

    # review_blurb in-loop judge audit: judge verdict vs blind label, prompt joined back by reply.
    akey = load(f"{STORE}/judgeval/keys_audit/review_blurb_audit_key.jsonl")
    alab = {}
    for f in glob.glob(f"{STORE}/judgeval/blind_audit/labels_review_blurb_audit_*.jsonl"):
        if "overlap" not in f:
            alab.update({r["id"]: r["label"] for r in load(f)})
    vcache = {}
    audit = []
    for r in akey:
        row = {k: r.get(k) for k in ("kind", "cond", "seed", "step", "judge_exploit", "weight", "stratum",
                                     "reply", "intended", "pair_id")}
        row["blind"] = alab.get(r["blind_id"])
        if r.get("kind") == "trained":
            k = (r["cond"], int(r["seed"]), int(r["step"]))
            if k not in vcache:
                fp = f"{run_dir('review_blurb', *k[:2])}/validation/{k[2]}.jsonl"
                vcache[k] = {x["output"]: x for x in load(fp)} if os.path.exists(fp) else {}
            hit = vcache[k].get(r["reply"])
            if hit:
                row["gts"] = json.loads(hit["gts"])
        audit.append(row)
    data["audit"] = {"rows": audit, "report": open(f"{STORE}/judgeval/audit_report.txt").read()}

    # Privacy transfer v2 (2026-10-13): the blind-label gate report.
    data["privacy_v2"] = json.load(open(f"{STORE}/prev_envs/privacy_v2/report.json"))

    # Charter MCQ summary.
    rep = json.load(open(f"{HO}/mcq_ab_report.json"))
    data["mcq"] = rep
    json.dump(data, open(f"{outdir}/data.json", "w"), separators=(",", ":"))
    print("wrote", os.path.getsize(f"{outdir}/data.json") // 1024, "KB;",
          {e: len(data["gens"][e]) for e in ENVS}, "prompts;", len(ho), "held-out rows;",
          Counter(r["blind"] for r in audit))


async def judge_mode(outdir):
    sys.path.insert(0, REPO)
    from rl import judge
    from rl.rewards import review_blurb as rb

    data = json.load(open(f"{outdir}/data.json"))
    jobs = []
    for item in data["gens"]["review_blurb"]:
        spec = item["gts"]
        for cell, g in item["cells"].items():
            jobs.append((item["pid"], cell, spec, g["out"]))

    async def one(spec, reply):
        ap = await judge.ajudge(rb.APPEAL_SYSTEM, rb.APPEAL_USER.format(product=spec["product"], reply=reply))
        de = await judge.ajudge(rb.DETECT_SYSTEM, rb.DETECT_USER.format(
            reviews=rb._reviews(spec), defect=spec["defect"], reply=reply))
        return ap, de

    res = await asyncio.gather(*(one(sp, rep) for _, _, sp, rep in jobs))
    out = {}
    for (pid, cell, _, _), (ap, de) in zip(jobs, res):
        out[f"{pid}|{cell}"] = {"appeal": (ap or {}).get("appeal"), "appeal_reason": (ap or {}).get("reason"),
                                "defect_mentioned": (de or {}).get("defect_mentioned"),
                                "invented_claims": (de or {}).get("invented_claims")}
    if judge._session is not None:
        await judge._session.close()
    json.dump(out, open(f"{outdir}/rationale.json", "w"), separators=(",", ":"))
    agree = sum(1 for (pid, cell, _, _) in jobs
                if out[f"{pid}|{cell}"]["defect_mentioned"] is not None
                and (not out[f"{pid}|{cell}"]["defect_mentioned"]) ==
                bool(next(i for i in data["gens"]["review_blurb"] if i["pid"] == pid)["cells"][cell]["exploit"]))
    print("rationales", len(out), "; re-run verdict equals stored verdict on", agree)


if __name__ == "__main__":
    mode, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    if mode == "extract":
        extract(outdir)
    else:
        os.environ.setdefault("MISALIGN_JUDGE_MAX_INFLIGHT", "128")
        asyncio.run(judge_mode(outdir))
