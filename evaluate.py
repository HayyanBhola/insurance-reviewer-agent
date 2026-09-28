"""
Regression gate: run this before every change you keep (and in CI on GitHub).

  python evaluate.py        -> free checks + latest AI run results, compare with thresholds
  python evaluate.py --ci   -> same, but skips checks whose inputs are missing (for GitHub Actions)

What it checks (all FREE, no API calls):
  1. Retrieval quality on the golden questions  (recall@3, MRR)       -> needs data/policy_index.json
  2. Decision accuracy of the rules-only pipeline on the 48 dev claims -> needs outputs/intake/*.json
  3. The latest AI run (outputs/specialists_summary_ai.json), if any:
     decision accuracy, faithfulness, cost per claim, p95 latency

Thresholds live in eval_thresholds.json. Every run is appended to outputs/eval_history.jsonl,
and it ALERTS if a metric dropped (or cost jumped) compared with the previous run.
Exit code 1 = a threshold failed, so CI marks the change as broken.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

THRESHOLDS = Path("eval_thresholds.json")
HISTORY = Path("outputs/eval_history.jsonl")
AI_SUMMARY = Path("outputs/specialists_summary_ai.json")

DEFAULT_THRESHOLDS = {
    "retrieval_recall_at_3_min": 0.90,
    "retrieval_mrr_min": 0.75,
    "rules_accuracy_min": 0.85,
    "ai_accuracy_min": 0.85,
    "ai_faithfulness_min": 0.90,
    "ai_cost_per_claim_max_usd": 0.02,
    "ai_p95_latency_max_s": 20,
    # alerts compare with the previous run
    "alert_drop": 0.05,          # a quality metric fell by more than 5 points
    "alert_cost_increase": 0.5,  # cost per claim rose by more than 50%
}


def rules_accuracy():
    """Run the three agents in rules-only mode on all dev claims (free)."""
    from data_access import build_claims_history
    from policy_index import INDEX_FILE, PolicyIndex
    from run_specialists import TRUTH, run_one

    index = PolicyIndex() if INDEX_FILE.exists() else None
    history = build_claims_history()
    ids = [c for c, t in TRUTH.items() if t["split"] == "dev"]
    done = correct = 0
    for cid in ids:
        try:
            r = run_one(cid, index, history, use_llm=False)
        except FileNotFoundError:
            continue
        done += 1
        correct += r.preliminary_decision == TRUTH[cid]["expected_decision"]
    return (correct / done if done else None), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true", help="skip checks whose inputs are missing")
    args = ap.parse_args()

    th = {**DEFAULT_THRESHOLDS,
          **(json.loads(THRESHOLDS.read_text(encoding="utf-8")) if THRESHOLDS.exists() else {})}
    results, checks, alerts = {}, [], []

    def check(name, value, ok, rule):
        checks.append((name, value, ok, rule))

    # 1. retrieval ----------------------------------------------------------------
    from policy_index import INDEX_FILE
    if INDEX_FILE.exists():
        from eval_retrieval import evaluate as eval_retrieval
        m = eval_retrieval(quiet=True)
        if m:
            results.update({"retrieval_recall_at_3": m["recall@3"], "retrieval_mrr": m["mrr"],
                            "retrieval_mode": m["mode"]})
            check("retrieval recall@3", m["recall@3"], m["recall@3"] >= th["retrieval_recall_at_3_min"],
                  f">= {th['retrieval_recall_at_3_min']}")
            check("retrieval MRR", m["mrr"], m["mrr"] >= th["retrieval_mrr_min"], f">= {th['retrieval_mrr_min']}")
    elif not args.ci:
        print("! No policy index. Run: python policy_index.py build --no-embed")

    # 2. rules-only pipeline ---------------------------------------------------------
    acc, n = rules_accuracy()
    if acc is not None:
        results["rules_accuracy"] = round(acc, 3)
        check(f"rules-only accuracy ({n} claims)", round(acc, 3), acc >= th["rules_accuracy_min"],
              f">= {th['rules_accuracy_min']}")
    elif not args.ci:
        print("! No intake results. Run: python run_intake.py --dev")

    # 3. latest AI run ---------------------------------------------------------------
    if AI_SUMMARY.exists():
        s = json.loads(AI_SUMMARY.read_text(encoding="utf-8"))
        results.update({"ai_model": s["model"], "ai_accuracy": s["accuracy"],
                        "ai_faithfulness": s["faithfulness"], "ai_cost_per_claim_usd": s["cost_per_claim_usd"],
                        "ai_p95_latency_s": s["p95_latency_s"]})
        check(f"AI accuracy ({s['model']})", s["accuracy"], s["accuracy"] >= th["ai_accuracy_min"],
              f">= {th['ai_accuracy_min']}")
        if s["faithfulness"] is not None:
            check("AI faithfulness", s["faithfulness"], s["faithfulness"] >= th["ai_faithfulness_min"],
                  f">= {th['ai_faithfulness_min']}")
        check("AI cost per claim ($)", s["cost_per_claim_usd"],
              s["cost_per_claim_usd"] <= th["ai_cost_per_claim_max_usd"], f"<= {th['ai_cost_per_claim_max_usd']}")
        if s["p95_latency_s"] is not None:
            check("AI p95 latency (s)", s["p95_latency_s"], s["p95_latency_s"] <= th["ai_p95_latency_max_s"],
                  f"<= {th['ai_p95_latency_max_s']}")

    # alerts: compare with the previous run -------------------------------------------
    previous = None
    if HISTORY.exists():
        lines = [line for line in HISTORY.read_text(encoding="utf-8").splitlines() if line.strip()]
        previous = json.loads(lines[-1])["results"] if lines else None
    if previous:
        for key in ("retrieval_recall_at_3", "retrieval_mrr", "rules_accuracy", "ai_accuracy", "ai_faithfulness"):
            a, b = previous.get(key), results.get(key)
            if a is not None and b is not None and a - b > th["alert_drop"]:
                alerts.append(f"{key} dropped from {a} to {b}")
        a, b = previous.get("ai_cost_per_claim_usd"), results.get("ai_cost_per_claim_usd")
        if a and b and (b - a) / a > th["alert_cost_increase"]:
            alerts.append(f"cost per claim jumped from ${a} to ${b}")

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "results": results}) + "\n")

    # report ------------------------------------------------------------------------
    print("\nEVALUATION GATE")
    for name, value, ok, rule in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} {value!s:<10} (needs {rule})")
    for a in alerts:
        print(f"  ALERT {a}")
    if not checks:
        print("  Nothing to check yet.")
    failed = [c for c in checks if not c[2]]
    print(f"\n{'FAILED' if failed else 'PASSED'}: {len(checks) - len(failed)}/{len(checks)} checks"
          + (f", {len(alerts)} alert(s)" if alerts else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
