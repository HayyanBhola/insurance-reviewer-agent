"""
Phase 3: run the coverage, evidence and fraud agents.

  python run_specialists.py C002                -> one claim, detailed report
  python run_specialists.py --dev --no-llm      -> all dev claims, rules only (FREE)
  python run_specialists.py --dev --limit 5     -> 5 dev claims with AI (a few cents)
  python run_specialists.py --dev               -> all 48 dev claims with AI

Needs:  outputs/intake/*.json   (from:  python run_intake.py --dev)
        data/policy_index.json  (from:  python policy_index.py build)

The "preliminary decision" here is a simple rule. Phase 4 replaces it with a
LangGraph workflow (decision agent, critic, human review).
"""

import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402

from coverage_agent import check_coverage  # noqa: E402
from data_access import build_claims_history, load_intake  # noqa: E402
from evidence_agent import check_evidence  # noqa: E402
from findings import SpecialistReport  # noqa: E402
from fraud_agent import check_fraud  # noqa: E402
from llm import CACHE_STATS, model_id  # noqa: E402
from metrics import cost_usd, faithfulness, percentile  # noqa: E402
from policy_index import INDEX_FILE, PolicyIndex  # noqa: E402

OUT = Path("outputs/specialists")
TRUTH = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))


def preliminary_decision(cov, ev, fr):
    """Simple, transparent rule. Deny only with a clear coverage reason."""
    if cov.status == "not_covered":
        # rules-based denials have one reason; RAG denials start with "Policy active..."
        why = cov.reasons[1:] if cov.method == "rules+rag" else cov.reasons
        why += [f'Policy says: "{c.quote}" ({c.policy_file}, p.{c.page})'
                for c in cov.citations if c.verified][:1]
        return "deny", why
    reasons = []
    if cov.status == "needs_review":
        reasons.append("Coverage needs human review: " + cov.reasons[-1])
    reasons += [f.message for f in ev.flags + fr.signals if f.severity in ("medium", "high")]
    if reasons:
        return "investigate", reasons
    return "approve", ["Covered, evidence consistent, no fraud signals."]


def run_one(cid, index, history, use_llm):
    intake = load_intake(cid)
    cov = check_coverage(intake, index, use_llm=use_llm)
    ev = check_evidence(intake, use_llm=use_llm)
    fr = check_fraud(cid, intake, history)
    decision, reasons = preliminary_decision(cov, ev, fr)
    return SpecialistReport(claim_id=cid, coverage=cov, evidence=ev, fraud=fr,
                            preliminary_decision=decision, decision_reasons=reasons)


def show(r):
    c, e, f = r.coverage, r.evidence, r.fraud
    print(f"\n=== {r.claim_id}: {r.preliminary_decision.upper()} ===")
    print(f"Coverage : {c.status} ({c.method}) | policy {c.coverage_type}, "
          f"active={c.policy_active_on_incident}, wording={c.policy_wording}")
    for reason in c.reasons:
        print(f"           - {reason}")
    for cit in c.citations:
        mark = "verified" if cit.verified else "NOT FOUND in policy"
        print(f'           > [{cit.chunk_id}] p.{cit.page} "{cit.quote[:110]}" ({mark})')
    rng = f"PKR {e.typical_range_pkr[0]:,.0f}-{e.typical_range_pkr[1]:,.0f}" if e.typical_range_pkr else "?"
    print(f"Evidence : photo = {e.photo_summary}")
    print(f"           cost {e.cost_status} (ratio {e.cost_ratio}, typical {rng}) | story consistent: "
          f"{e.description_consistent}")
    print(f"           {e.description_note}")
    print(f"Fraud    : risk {f.risk}")
    for s in f.signals:
        print(f"           - [{s.severity}] {s.code}: {s.message}")
    print("Why      : " + " | ".join(r.decision_reasons))


def report(rows, n_total, seconds, usage, reports=(), latencies=(), mode="ai"):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    print(f"\nProcessed {n}/{n_total} claims in {seconds:.0f}s")
    print(f"Decision accuracy: {correct}/{n} = {correct / n:.0%}")

    labels = ["approve", "investigate", "deny"]
    cm = Counter((r["expected"], r["predicted"]) for r in rows)
    print("\nConfusion matrix (rows = correct answer, columns = system):")
    print(f"{'':>13}" + "".join(f"{l:>13}" for l in labels))
    for exp in labels:
        print(f"{exp:>13}" + "".join(f"{cm[(exp, pred)]:>13}" for pred in labels))

    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r["correct"])
    print("\nBy scenario (how many the system got right):")
    for sc, oks in sorted(by_scenario.items()):
        print(f"  {sc:<20} {sum(oks)}/{len(oks)}")

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print("\nWrong decisions:")
        for r in wrong:
            print(f"  {r['claim_id']} {r['scenario']:<19} expected {r['expected']:<11} got "
                  f"{r['predicted']:<11} {r['why'][:90]}")

    # --- quality and operations metrics -------------------------------------
    print("\nOperations")
    rate, ok, total = faithfulness(reports)
    print(f"  Faithfulness (policy quotes verified): "
          + (f"{ok}/{total} = {rate:.0%}" if total else "n/a (no AI citations in this run)"))
    if latencies:
        print(f"  Latency per claim: p50 {percentile(latencies, 50):.2f}s | p95 {percentile(latencies, 95):.2f}s")
    tin = sum(u.get("input_tokens", 0) for u in usage.values())
    tout = sum(u.get("output_tokens", 0) for u in usage.values())
    dollars, unpriced = cost_usd(usage)
    if usage:
        cost_txt = f"${dollars:.4f} total, ${dollars / n:.5f} per claim" if dollars is not None else "unknown price"
        print(f"  Tokens: {tin:,} in + {tout:,} out ({', '.join(usage)}) | cost {cost_txt}"
              + (f" | no price for {unpriced}" if unpriced else ""))
    else:
        why = ("rules-only run" if mode == "rules" else
               "all answers from cache" if CACHE_STATS["misses"] == 0 else "no token data reported")
        print(f"  Tokens: 0 | cost $0 ({why})")
    print(f"  LLM cache: {CACHE_STATS['hits']} hits, {CACHE_STATS['misses']} new calls")

    # save a summary for the regression gate (evaluate.py)
    summary = {"mode": mode, "model": model_id() if mode == "ai" else "rules", "claims": n,
               "accuracy": round(correct / n, 3),
               "faithfulness": round(rate, 3) if rate is not None else None,
               "p95_latency_s": round(percentile(latencies, 95), 3) if latencies else None,
               "cost_per_claim_usd": round(dollars / n, 6) if usage and dollars is not None else 0.0,
               "tokens_in": tin, "tokens_out": tout,
               "by_scenario": {sc: f"{sum(o)}/{len(o)}" for sc, o in by_scenario.items()}}
    Path("outputs").mkdir(exist_ok=True)
    Path(f"outputs/specialists_summary_{mode}.json").write_text(json.dumps(summary, indent=2),
                                                                 encoding="utf-8")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/specialists_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Per-claim results: outputs/specialists_report.csv  |  full reports: {OUT}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("claim_id", nargs="?")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-llm", action="store_true", help="rules only, no API calls (free)")
    args = ap.parse_args()
    use_llm = not args.no_llm

    index = PolicyIndex() if INDEX_FILE.exists() else None
    if index is None:
        print(f"Note: {INDEX_FILE} not found, so policy clauses can't be checked. "
              f"Run: python policy_index.py build")
    history = build_claims_history()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.claim_id:
        cid = args.claim_id.upper()
        if TRUTH.get(cid, {}).get("split") == "test":
            sys.exit(f"{cid} is a TEST claim. Keep it for the final evaluation.")
        r = run_one(cid, index, history, use_llm)
        (OUT / f"{cid}.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
        show(r)
        t = TRUTH[cid]
        print(f"\nCorrect answer: {t['expected_decision']} ({t['scenario']})")
        return

    if not args.dev:
        ap.print_help()
        return

    ids = [c for c, t in TRUTH.items() if t["split"] == "dev"][: args.limit]
    rows, reports, latencies, start = [], [], [], time.time()
    with get_usage_metadata_callback() as cb:
        for cid in ids:
            try:
                t0 = time.perf_counter()
                r = run_one(cid, index, history, use_llm)
                latencies.append(time.perf_counter() - t0)
                reports.append(r)
            except FileNotFoundError as exc:
                print(f"{cid}: SKIPPED ({exc})")
                continue
            except Exception as exc:
                print(f"{cid}: FAILED ({type(exc).__name__}: {exc})")
                continue
            (OUT / f"{cid}.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
            t = TRUTH[cid]
            ok = r.preliminary_decision == t["expected_decision"]
            rows.append({"claim_id": cid, "scenario": t["scenario"], "expected": t["expected_decision"],
                         "predicted": r.preliminary_decision, "correct": ok,
                         "coverage": r.coverage.status, "cost_status": r.evidence.cost_status,
                         "story_consistent": r.evidence.description_consistent,
                         "fraud_risk": r.fraud.risk, "why": " | ".join(r.decision_reasons)})
            print(f"{cid}: {r.preliminary_decision:<11} (expected {t['expected_decision']:<11}) "
                  f"{'OK' if ok else 'X '} {t['scenario']}")
        usage = dict(cb.usage_metadata)

    if rows:
        report(rows, len(ids), time.time() - start, usage, reports, latencies,
               mode="ai" if use_llm else "rules")


if __name__ == "__main__":
    main()
