"""
Run the Phase 4 LangGraph workflow.

  python run_graph.py C002                     -> run until it PAUSES for the adjuster (free, rules-only)
  python run_graph.py C002 --llm               -> same, but the AI writes the decision note
  python run_graph.py C002 --resume accept     -> adjuster accepts the recommendation
  python run_graph.py C002 --resume deny --note "Receipt looks forged"   -> adjuster overrides
  python run_graph.py C002 --status            -> where is this claim? (saved state + audit trail)
  python run_graph.py --dev [--limit 5] [--llm] -> all dev claims, auto-accepting, with a score
  python run_graph.py --draw                    -> save the graph diagram (Mermaid) to outputs/
  python run_graph.py --dev --llm --save NAME   -> also save the 48 decisions to outputs/graph_runs/NAME.json
  python run_graph.py --compare-runs OLD NEW    -> compare two saved runs with the rule decided in advance

Paused claims are saved in outputs/checkpoints.sqlite, so you can close the terminal
and resume a claim later: that is what the checkpointer is for.
"""

import argparse
import json
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from claim_graph import build_graph  # noqa: E402
from metrics import cost_usd, percentile  # noqa: E402

TRUTH = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
CHECKPOINTS = Path("outputs/checkpoints.sqlite")
LABELS = {"intake": "Reading documents", "request_info": "Waiting for missing documents",
          "coverage": "Coverage agent", "evidence": "Evidence agent", "fraud": "Fraud agent",
          "decide": "Decision agent", "critic": "Critic", "human_review": "Adjuster review",
          "finalize": "Finalising"}


def persistent_graph():
    CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINTS, check_same_thread=False)  # parallel nodes use threads
    return build_graph(SqliteSaver(conn))


def config_for(cid, use_llm):
    return {"configurable": {"thread_id": cid, "use_llm": use_llm}}


def stream(graph, inp, cfg, quiet=False):
    """Run the graph, printing each step live. Returns the interrupt payload if it paused."""
    paused = None
    for update in graph.stream(inp, cfg, stream_mode="updates"):
        for node, change in update.items():
            if node == "__interrupt__":
                paused = change[0].value
                continue
            if not quiet and change and change.get("audit"):
                print(f"  [{LABELS.get(node, node)}] {change['audit'][-1]['summary']}")
    return paused


def show_pause(cid, p):
    print(f"\n=== {cid} is WAITING FOR A HUMAN ===")
    if p["type"] == "adjuster_review":
        print(f"Recommendation: {p['recommendation'].upper()}"
              + (f"  (AI escalated from approve: {p['escalation_reason']})" if p.get("escalated_by_ai") else ""))
        print(f"Summary:        {p['summary']}")
        print(f"Justification:  {p['justification']}")
        print(f"\nDecide with:  python run_graph.py {cid} --resume accept"
              f"   (or approve / deny / investigate, optionally --note \"...\")")
    else:
        print(f"Documents incomplete. Missing: {p['missing_fields']} | problems: {p['issues']}")
        print(f"\nDecide with:  python run_graph.py {cid} --resume continue   (or close)")


def show_final(state):
    print(f"\n=== {state['claim_id']}: {state.get('status', '?').upper()} ===")
    if state.get("final_decision"):
        h = state["human"]
        print(f"Final decision: {state['final_decision'].upper()} "
              f"(recommended {state['recommendation']}"
              + (f", OVERRIDDEN by {h['reviewer']}" if h["overridden"] else ", accepted") + ")")
        if h.get("note"):
            print(f"Adjuster note:  {h['note']}")


def show_status(graph, cid):
    snap = graph.get_state(config_for(cid, False))
    if not snap.values:
        print(f"No saved run for {cid}.")
        return
    waiting = [t.name for t in snap.tasks if t.interrupts]
    print(f"{cid}: " + (f"PAUSED at {waiting[0]}" if waiting else f"status {snap.values.get('status', '?')}"))
    print("Audit trail:")
    for a in snap.values.get("audit", []):
        print(f"  {a['time']}  {a['node']:<13} {a['summary']}")


def why(s):
    """Which agent drove the recommendation: shown for every miss, so you know where to look."""
    parts = []
    cov = s.get("coverage", {})
    if cov.get("status") != "covered":
        parts.append(f"coverage {cov.get('status')} ({cov.get('method')}): {(cov.get('reasons') or [''])[-1][:110]}")
    for f in s.get("evidence", {}).get("flags", []) + s.get("fraud", {}).get("signals", []):
        if f["severity"] in ("medium", "high"):
            parts.append(f"{f['code']} ({f['severity']}): {f['message'][:110]}")
    if s.get("recommendation") != s.get("rule_recommendation"):
        parts.append(f"AI escalated: {s.get('writeup', {}).get('escalation_reason', '')[:110]}")
    return "\n            ".join(parts) or "no flags (rules recommended approve)"


RUNS = Path("outputs/graph_runs")
FRAUD_SCENARIOS = {"exaggerated_damage", "inflated_estimate", "reused_photo", "duplicate_invoice",
                   "new_policy", "lapsed_policy", "third_party_only"}


def keep_rule(old, new):
    """Decided BEFORE seeing the new run. A change is kept only if ALL are true.
    old/new: {claim_id: {"scenario", "expected", "got"}} over the same claims."""
    def correct(rows, pick=lambda r: True):
        return sum(r["got"] == r["expected"] for r in rows.values() if pick(r))

    def money_out(rows):  # approved although it should not be: the costly mistake
        return sum(r["got"] == "approve" and r["expected"] != "approve" for r in rows.values())

    checks = [("honest claims: more correct",
               correct(new, lambda r: r["scenario"] == "honest") > correct(old, lambda r: r["scenario"] == "honest"))]
    for sc in sorted(FRAUD_SCENARIOS):
        checks.append((f"{sc}: not worse",
                       correct(new, lambda r, sc=sc: r["scenario"] == sc) >= correct(old, lambda r, sc=sc: r["scenario"] == sc)))
    checks.append(("excluded_tyre: not worse",
                   correct(new, lambda r: r["scenario"] == "excluded_tyre") >= correct(old, lambda r: r["scenario"] == "excluded_tyre")))
    checks.append(("no more wrong approvals (money out)", money_out(new) <= money_out(old)))
    return all(ok for _, ok in checks), checks


def _load_run(name):
    path = RUNS / f"{name}.json"
    if not path.exists():
        saved = sorted(p.stem for p in RUNS.glob("*.json")) if RUNS.exists() else []
        raise SystemExit(f"No saved run called '{name}'. Saved runs: {saved or 'none'}")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def keep_rule_money_out(old, new):
    """For TYRE_REVIEW (decided before the run): fewer wrong approvals, no fraud scenario worse,
    at most 2 more honest claims held up."""
    def money_out(rows):
        return sum(r["got"] == "approve" and r["expected"] != "approve" for r in rows.values())

    def correct(rows, sc):
        return sum(r["got"] == r["expected"] for r in rows.values() if r["scenario"] == sc)

    def held(rows):
        return sum(r["scenario"] == "honest" and r["got"] != "approve" for r in rows.values())

    checks = [("fewer wrong approvals (money out)", money_out(new) < money_out(old))]
    for sc in sorted(FRAUD_SCENARIOS):
        checks.append((f"{sc}: not worse", correct(new, sc) >= correct(old, sc)))
    checks.append((f"honest held up: at most 2 more ({held(old)} -> {held(new)})", held(new) - held(old) <= 2))
    return all(ok for _, ok in checks), checks


RULES = {"honest": keep_rule, "money_out": keep_rule_money_out}


def compare_runs(old_name, new_name, rule="honest"):
    old, new = _load_run(old_name), _load_run(new_name)
    dev = {c for c, t in TRUTH.items() if t["split"] == "dev"}
    missing = {name: sorted(dev - set(rows)) for name, rows in ((old_name, old), (new_name, new))}
    if any(missing.values()):
        # a claim that failed could be exactly the one a change breaks: no verdict on a partial run
        for name, m in missing.items():
            if m:
                print(f"Run '{name}' is missing {len(m)} dev claim(s): {m}")
        raise SystemExit("No verdict: both runs must cover all dev claims. Re-run the incomplete one "
                         "(answers that already came back are cached, so it is cheap).")
    common = sorted(dev)
    old, new = {c: old[c] for c in common}, {c: new[c] for c in common}
    print(f"\n{'scenario':<20}{old_name:>16}{new_name:>16}")
    for sc in sorted({r["scenario"] for r in old.values()}):
        a = [r for r in old.values() if r["scenario"] == sc]
        b = [r for r in new.values() if r["scenario"] == sc]
        print(f"{sc:<20}{sum(r['got'] == r['expected'] for r in a):>13}/{len(a):<2}"
              f"{sum(r['got'] == r['expected'] for r in b):>13}/{len(b):<2}")
    print(f"{'TOTAL':<20}{sum(r['got'] == r['expected'] for r in old.values()):>13}/{len(old):<2}"
          f"{sum(r['got'] == r['expected'] for r in new.values()):>13}/{len(new):<2}")
    changed = [c for c in common if old[c]["got"] != new[c]["got"]]
    if changed:
        print("\nChanged decisions:")
        for c in changed:
            mark = "fixed" if new[c]["got"] == new[c]["expected"] else (
                "BROKE" if old[c]["got"] == old[c]["expected"] else "still wrong")
            print(f"  {c} {old[c]['scenario']:<19} expected {old[c]['expected']:<11} "
                  f"{old[c]['got']:>11} -> {new[c]['got']:<11} {mark}")
    keep, checks = RULES[rule](old, new)
    print(f"\nRule '{rule}', decided in advance (all must pass):")
    for text, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    print(f"\nVERDICT: {'KEEP the change' if keep else 'REJECT the change'}")
    return keep


def run_batch(limit, use_llm, save=None):
    import os
    from llm import checks_version
    from llm import ai_escalation, tyre_review
    versions = {"ai_checks": checks_version() if use_llm else "rules-only",
                "photo_prompt": os.getenv("PHOTO_PROMPT", "v1"),
                "ai_escalation": ai_escalation(), "tyre_review": tyre_review()}
    print(f"AI checks: {versions['ai_checks']} | photo prompt: {versions['photo_prompt']} | "
          f"AI escalation: {'on' if versions['ai_escalation'] else 'off'} | "
          f"tyre review: {'on' if versions['tyre_review'] else 'off'}")
    graph = build_graph(InMemorySaver())
    ids = [c for c, t in TRUTH.items() if t["split"] == "dev"][:limit]
    rows, lat, start = [], [], time.time()
    with get_usage_metadata_callback() as cb:
        for cid in ids:
            cfg = config_for(f"batch-{cid}", use_llm)
            t0 = time.perf_counter()
            try:
                paused = stream(graph, {"claim_id": cid}, cfg, quiet=True)
                while paused:  # auto-answer every pause: accept / continue
                    answer = {"action": "accept" if paused["type"] == "adjuster_review" else "continue",
                              "reviewer": "auto"}
                    paused = stream(graph, Command(resume=answer), cfg, quiet=True)
            except Exception as exc:
                print(f"{cid}: FAILED ({type(exc).__name__}: {exc})")
                continue
            lat.append(time.perf_counter() - t0)
            s = graph.get_state(cfg).values
            exp = TRUTH[cid]["expected_decision"]
            rows.append({"cid": cid, "scenario": TRUTH[cid]["scenario"], "expected": exp,
                         "got": s.get("final_decision"), "revisions": s.get("revisions", 0),
                         "template": s.get("used_template"), "critic_ok": s["critic"]["passed"]})
            r = rows[-1]
            print(f"{cid}: {str(r['got']):<11} (expected {exp:<11}) {'OK' if r['got'] == exp else 'X '} "
                  f"| writer attempts {r['revisions']}{' (template)' if r['template'] else ''}")
            if r["got"] != exp:
                print(f"       why: {why(s)}")
        usage = dict(cb.usage_metadata)

    n = len(rows)
    if not n:
        return
    if save:
        RUNS.mkdir(parents=True, exist_ok=True)
        complete = n == len(ids)
        (RUNS / f"{save}.json").write_text(json.dumps({
            "label": save, "use_llm": use_llm, "models": list(usage), "complete": complete, **versions,
            "rows": {r["cid"]: {k: r[k] for k in ("scenario", "expected", "got")} for r in rows}},
            indent=1), encoding="utf-8")
        print(f"Saved {n} decisions to {RUNS / (save + '.json')}"
              + ("" if complete else "  (INCOMPLETE: some claims failed)"))
    ok = sum(r["got"] == r["expected"] for r in rows)
    print(f"\nProcessed {n} claims through the graph in {time.time() - start:.0f}s")
    print(f"Decision accuracy: {ok}/{n} = {ok / n:.0%}")
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r["got"] == r["expected"])
    for sc, oks in sorted(by.items()):
        print(f"  {sc:<20} {sum(oks)}/{len(oks)}")
    first_pass = sum(1 for r in rows if r["revisions"] == 1)
    print(f"Critic: passed on first write {first_pass}/{n}; template fallbacks "
          f"{sum(1 for r in rows if r['template'] and use_llm)}")
    print(f"Latency per claim: p50 {percentile(lat, 50):.2f}s | p95 {percentile(lat, 95):.2f}s")
    dollars, _ = cost_usd(usage)
    tin = sum(u.get("input_tokens", 0) for u in usage.values())
    tout = sum(u.get("output_tokens", 0) for u in usage.values())
    print(f"Tokens: {tin:,} in + {tout:,} out" + (f" | ${dollars:.4f}" if dollars else " | $0"))
    if usage:
        print("Models used: " + ", ".join(usage))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("claim_id", nargs="?")
    ap.add_argument("--llm", action="store_true", help="let the AI write the decision note")
    ap.add_argument("--resume", help="accept / approve / deny / investigate / continue / close")
    ap.add_argument("--note", default="")
    ap.add_argument("--reviewer", default="adjuster")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--save", help="with --dev: save the decisions under this name")
    ap.add_argument("--compare-runs", nargs=2, metavar=("OLD", "NEW"))
    ap.add_argument("--rule", choices=["honest", "money_out"], default="honest",
                    help="which keep rule --compare-runs applies")
    args = ap.parse_args()

    if args.draw:
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/claim_graph.mmd").write_text(build_graph().get_graph().draw_mermaid(), encoding="utf-8")
        print("Saved outputs/claim_graph.mmd (paste it into https://mermaid.live to see the diagram)")
        return
    if args.compare_runs:
        compare_runs(*args.compare_runs, rule=args.rule)
        return
    if args.dev:
        run_batch(args.limit, args.llm, save=args.save)
        return
    if not args.claim_id:
        ap.print_help()
        return

    cid = args.claim_id.upper()
    if TRUTH.get(cid, {}).get("split", "dev") != "dev":
        sys.exit(f"{cid} is a TEST claim. Keep it for the final evaluation.")
    graph = persistent_graph()
    cfg = config_for(cid, args.llm)

    if args.status:
        show_status(graph, cid)
        return

    snap = graph.get_state(cfg)
    waiting = snap.values and any(t.interrupts for t in snap.tasks)
    if args.resume:
        if not waiting:
            sys.exit(f"{cid} is not waiting for a decision. Run: python run_graph.py {cid}")
        print(f"Resuming {cid} with '{args.resume}'...")
        paused = stream(graph, Command(resume={"action": args.resume, "note": args.note,
                                               "reviewer": args.reviewer}), cfg)
    else:
        if waiting:
            print(f"{cid} is already paused and waiting. Showing the saved request:")
            show_pause(cid, snap.tasks[0].interrupts[0].value)
            return
        print(f"Processing {cid} ({'AI write-up' if args.llm else 'rules-only, free'})...")
        paused = stream(graph, {"claim_id": cid}, cfg)

    if paused:
        show_pause(cid, paused)
    else:
        show_final(graph.get_state(cfg).values)


if __name__ == "__main__":
    main()
