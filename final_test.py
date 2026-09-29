"""
FINAL TEST: the whole system, once, on the 10 test claims nobody has looked at.

  python final_test.py                  -> checks every setting and shows the plan (no AI calls, free)
  python final_test.py --go             -> runs it: intake (forms + photos) then the full AI graph
  python final_test.py --show           -> shows the saved result again
  add  --set test2  for the second held-out set (20 claims, C059-C078)
  add  --set test3  for the third (20 claims, C079-C098, same system as test2)
  python final_test.py --combined test2 test3   -> one summary over both

Why only once
  Every improvement so far was chosen by looking at the 48 dev claims, so their score (92%)
  is optimistic. These 10 claims were never looked at. Their score is the honest one, but
  only if nothing is changed after seeing it. So the result is saved permanently in
  outputs/final_test.json with a fingerprint of the code and prompts, and a second full
  run is refused. (If a run stops half way, --go finishes the missing claims.)

Settings (same as the dev runs, and as recorded in outputs/policy_decisions.jsonl)
  intake: forms read with gpt-5.4-mini (as in Phase 2), photos with gpt-5.4-mini + prompt v3
  graph:  AI checks v2, text agents on gpt-5.4-nano
"""

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

DECISIONS = Path("outputs/policy_decisions.jsonl")
TRUTH = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))

# Each held-out set is run once, with the settings that were frozen before it.
SETS = {
    "test": {"result": Path("outputs/final_test.json"), "required": {
        "LLM_PROVIDER": "openai", "AI_CHECKS": "v2", "PHOTO_PROMPT": "v3",
        "OPENAI_VISION_MODEL": "gpt-5.4-mini"}},
    "test2": {"result": Path("outputs/final_test2.json"), "required": {
        "LLM_PROVIDER": "openai", "AI_CHECKS": "v2", "PHOTO_PROMPT": "v3",
        "OPENAI_VISION_MODEL": "gpt-5.4-mini", "AI_ESCALATION": "off", "TYRE_REVIEW": "on"},
        "labels": Path("data/labels_test2.csv")},
    # test3: the SAME frozen system as test2 (same code fingerprint), on the 19 unused photos
    "test3": {"result": Path("outputs/final_test3.json"), "required": {
        "LLM_PROVIDER": "openai", "AI_CHECKS": "v2", "PHOTO_PROMPT": "v3",
        "OPENAI_VISION_MODEL": "gpt-5.4-mini", "AI_ESCALATION": "off", "TYRE_REVIEW": "on"},
        "labels": Path("data/labels_test2.csv"), "same_system_as": "test2"},
}
SET = "test"
RESULT = SETS[SET]["result"]
REQUIRED = SETS[SET]["required"]
TEST_IDS = sorted(c for c, t in TRUTH.items() if t["split"] == SET)


def use_set(name):
    """Switch every module-level setting to one held-out set."""
    global SET, RESULT, REQUIRED, TEST_IDS
    SET, RESULT, REQUIRED = name, SETS[name]["result"], SETS[name]["required"]
    TEST_IDS = sorted(c for c, t in TRUTH.items() if t["split"] == name)
INTAKE_MODEL = "gpt-5.4-mini"
GRAPH_MODEL = "gpt-5.4-nano"
CODE_FILES = ["intake.py", "coverage_agent.py", "evidence_agent.py", "fraud_agent.py",
              "decision_agent.py", "claim_graph.py", "policy_index.py", "schemas.py", "findings.py"]


def fingerprint():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:12]


def preflight():
    """Returns a list of problems (empty = ready)."""
    from dotenv import load_dotenv
    load_dotenv()
    problems = []
    defaults = {"AI_CHECKS": "v2", "PHOTO_PROMPT": "v1", "LLM_PROVIDER": "gemini",
                "AI_ESCALATION": "on", "TYRE_REVIEW": "off"}
    for key, want in REQUIRED.items():
        have = os.getenv(key, defaults.get(key, ""))
        if have != want:
            problems.append(f"{key} is '{have}', must be '{want}' (set it in .env)")
    if not DECISIONS.exists():
        problems.append("outputs/policy_decisions.jsonl is missing (the decision must be recorded first)")
    if not Path("data/policy_index.json").exists():
        problems.append("policy index missing: run  python policy_index.py build")
    if not TEST_IDS:
        problems.append(f"no claims with split '{SET}' (for test2 run: python generate_test2.py)")
    missing = [c for c in TEST_IDS if not (Path("data/claims") / c).exists()]
    if missing:
        problems.append(f"claim folders missing: {missing}")
    return problems


def load_result():
    return json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else None


def save_result(data):
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def summarize(rows):
    n = len(rows)
    ok = sum(r["got"] == r["expected"] for r in rows)
    return {
        "claims": n,
        "correct": ok,
        "accuracy": round(ok / n, 3) if n else None,
        "wrong_approvals": sum(r["got"] == "approve" and r["expected"] != "approve" for r in rows),
        "wrong_denials": sum(r["got"] == "deny" and r["expected"] != "deny" for r in rows),
        "honest_held_up": sum(r["scenario"] == "honest" and r["got"] != "approve" for r in rows),
        "photo_type_correct": sum(r["photo_type_ok"] for r in rows),
        "photo_severity_within_one": sum(r["photo_within_one"] for r in rows),
    }


def report(data):
    s = data["summary"]
    print(f"\nFINAL TEST ({data.get('set', 'test')}) | {data['time']} | code fingerprint {data['fingerprint']}")
    print(f"settings: {data['settings']}")
    print(f"\n{'claim':<6}{'scenario':<20}{'expected':<12}{'got':<12}")
    for r in data["rows"]:
        mark = "OK" if r["got"] == r["expected"] else "X "
        print(f"{r['cid']:<6}{r['scenario']:<20}{r['expected']:<12}{str(r['got']):<12}{mark}")
        if r["got"] != r["expected"]:
            print(f"{'':<8}why: {r['why']}")
    print(f"\nAccuracy:            {s['correct']}/{s['claims']} = {s['accuracy']:.0%}")
    print(f"Wrong approvals:     {s['wrong_approvals']}   (money paid that should not be)")
    print(f"Wrong denials:       {s['wrong_denials']}")
    print(f"Honest held up:      {s['honest_held_up']}")
    print(f"Photo type correct:  {s['photo_type_correct']}/{s['claims']} | severity within one: "
          f"{s['photo_severity_within_one']}/{s['claims']}")
    print(f"Cost: ${data['cost_usd']:.4f} | {data['tokens_in']:,} in + {data['tokens_out']:,} out")
    if not data.get("complete"):
        print("\n!! INCOMPLETE: run  python final_test.py --go  again to finish the missing claims.")


def run():
    from langchain_core.callbacks import get_usage_metadata_callback
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    import csv
    import eval_photos
    from claim_graph import build_graph
    from intake import run_intake
    from metrics import cost_usd
    from run_graph import stream, why
    from run_intake import save as save_intake

    if SETS[SET].get("labels"):  # photo scoring needs this set's own frozen labels
        with open(SETS[SET]["labels"], newline="", encoding="utf-8") as f:
            eval_photos.LABELS.update({r["photo"]: r for r in csv.DictReader(f)})
    previous = load_result()
    if previous and previous.get("complete"):
        print("The final test has already been run. Its result is final, so it is not run again.")
        report(previous)
        return
    fp = fingerprint()
    twin = SETS[SET].get("same_system_as")
    if twin and SETS[twin]["result"].exists():
        twin_fp = json.loads(SETS[twin]["result"].read_text(encoding="utf-8"))["fingerprint"]
        if twin_fp != fp:
            sys.exit(f"The code changed since {twin} ({twin_fp} -> {fp}). {SET} must test the SAME system, "
                     "so it is refused.")
    if previous and previous["fingerprint"] != fp:
        sys.exit("The code changed since the half-finished final test started. That would mix two "
                 "versions in one result, so it is refused.")

    done = {r["cid"]: r for r in (previous or {}).get("rows", [])}
    todo = [c for c in TEST_IDS if c not in done]
    print(f"Final test ({SET}) on {len(TEST_IDS)} held-out claims ({len(done)} already done, {len(todo)} to go)")
    graph = build_graph(InMemorySaver())
    rows = list(done.values())
    t0 = time.time()
    with get_usage_metadata_callback() as cb:
        for cid in todo:
            t = TRUTH[cid]
            try:
                # 1. intake: forms with mini (as in Phase 2), photo with mini + prompt v3
                os.environ["OPENAI_MODEL"] = INTAKE_MODEL
                path = Path(f"outputs/intake/{cid}.json")
                if not path.exists():
                    save_intake(run_intake(cid))
                # 2. the full AI graph with checks v2 on nano, adjuster auto-accepts
                os.environ["OPENAI_MODEL"] = GRAPH_MODEL
                cfg = {"configurable": {"thread_id": f"final-{cid}", "use_llm": True}}
                paused = stream(graph, {"claim_id": cid}, cfg, quiet=True)
                while paused:
                    answer = {"action": "accept" if paused["type"] == "adjuster_review" else "continue",
                              "reviewer": "auto"}
                    paused = stream(graph, Command(resume=answer), cfg, quiet=True)
                s = graph.get_state(cfg).values
            except Exception as exc:  # noqa: BLE001
                print(f"{cid}: FAILED ({type(exc).__name__}: {str(exc)[:150]})")
                continue
            photo = json.loads(path.read_text(encoding="utf-8"))["photo"]
            ps = eval_photos.score(t["photo_source"], photo) if t["photo_source"] in eval_photos.LABELS else {}
            row = {"cid": cid, "scenario": t["scenario"], "expected": t["expected_decision"],
                   "got": s.get("final_decision"), "why": why(s) if s.get("final_decision") != t["expected_decision"] else "",
                   "photo_type_ok": bool(ps.get("type_ok")), "photo_within_one": bool(ps.get("within_one"))}
            rows.append(row)
            print(f"{cid}: {row['got']:<11} (expected {row['expected']})")
        usage = dict(cb.usage_metadata)

    rows.sort(key=lambda r: r["cid"])
    dollars, _ = cost_usd(usage)
    prev_cost = (previous or {}).get("cost_usd", 0.0)
    data = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "set": SET,
        "fingerprint": fp,
        "settings": {**REQUIRED, "intake_model": INTAKE_MODEL, "graph_model": GRAPH_MODEL},
        "complete": len(rows) == len(TEST_IDS),
        "rows": rows,
        "summary": summarize(rows),
        "cost_usd": round(prev_cost + (dollars or 0.0), 4),
        "tokens_in": (previous or {}).get("tokens_in", 0) + sum(u.get("input_tokens", 0) for u in usage.values()),
        "tokens_out": (previous or {}).get("tokens_out", 0) + sum(u.get("output_tokens", 0) for u in usage.values()),
        "seconds": round((previous or {}).get("seconds", 0) + time.time() - t0),
    }
    save_result(data)
    report(data)
    if data["complete"]:
        print(f"\nSaved to {RESULT}. Commit it: this is your final, honest result.")
    else:
        print(f"\nPartial result saved to {RESULT}. Run  python final_test.py --go  again to finish.")


def combined(names):
    """One summary over several finished sets that tested the same system."""
    datas = []
    for n in names:
        p = SETS[n]["result"]
        if not p.exists():
            sys.exit(f"{n} has not been run yet.")
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("complete"):
            sys.exit(f"{n} is not complete.")
        datas.append(d)
    fps = {d["fingerprint"] for d in datas}
    if len(fps) > 1:
        sys.exit(f"These sets tested different code ({fps}); they cannot be combined.")
    rows = [r for d in datas for r in d["rows"]]
    s = summarize(rows)
    import math
    n, k = s["claims"], s["correct"]
    z = 1.96
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    print(f"COMBINED {' + '.join(names)} | same code fingerprint {fps.pop()}")
    print(f"Accuracy:        {k}/{n} = {k / n:.0%}   (95% range {centre - half:.0%} - {centre + half:.0%})")
    print(f"Wrong approvals: {s['wrong_approvals']} | wrong denials: {s['wrong_denials']} | "
          f"honest held up: {s['honest_held_up']}")
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r["got"] == r["expected"])
    for sc, oks in sorted(by.items()):
        print(f"  {sc:<20} {sum(oks)}/{len(oks)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="run the final test")
    ap.add_argument("--show", action="store_true", help="show the saved result")
    ap.add_argument("--combined", nargs="+", metavar="SET", help="e.g. --combined test2 test3")
    ap.add_argument("--set", choices=sorted(SETS), default="test",
                    help="which held-out set: test (first, already run) or test2 (new)")
    args = ap.parse_args()
    if args.combined:
        combined(args.combined)
        return
    use_set(args.set)

    if args.show:
        data = load_result()
        report(data) if data else print("The final test has not been run yet.")
        return
    problems = preflight()
    print(f"Test claims: {', '.join(TEST_IDS)}")
    print(f"Intake: forms {INTAKE_MODEL}, photos {REQUIRED['OPENAI_VISION_MODEL']} + prompt "
          f"{REQUIRED['PHOTO_PROMPT']} | graph: checks {REQUIRED['AI_CHECKS']} on {GRAPH_MODEL}")
    print(f"Code fingerprint: {fingerprint()}")
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    if not args.go:
        previous = load_result()
        if previous and previous.get("complete"):
            print(f"\nAlready run. See the result with:  python final_test.py --set {SET} --show")
        else:
            print(f"\nREADY. Estimated cost 10-40 cents. Run it once with:  python final_test.py --set {SET} --go")
        return
    run()


if __name__ == "__main__":
    main()
