"""
Run the Phase 4 LangGraph workflow.

  python run_graph.py C002                     -> run until it PAUSES for the adjuster (free, rules-only)
  python run_graph.py C002 --llm               -> same, but the AI writes the decision note
  python run_graph.py C002 --resume accept     -> adjuster accepts the recommendation
  python run_graph.py C002 --resume deny --note "Receipt looks forged"   -> adjuster overrides
  python run_graph.py C002 --status            -> where is this claim? (saved state + audit trail)
  python run_graph.py --dev [--limit 5] [--llm] -> all dev claims, auto-accepting, with a score
  python run_graph.py --draw                    -> save the graph diagram (Mermaid) to outputs/

Paused claims are saved in outputs/checkpoints.sqlite, so you can close the terminal
and resume a claim later: that is what the checkpointer is for.
"""

import argparse
import json
import sqlite3
import sys
import time
import warnings
from collections import Counter
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


def run_batch(limit, use_llm):
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
    args = ap.parse_args()

    if args.draw:
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/claim_graph.mmd").write_text(build_graph().get_graph().draw_mermaid(), encoding="utf-8")
        print("Saved outputs/claim_graph.mmd (paste it into https://mermaid.live to see the diagram)")
        return
    if args.dev:
        run_batch(args.limit, args.llm)
        return
    if not args.claim_id:
        ap.print_help()
        return

    cid = args.claim_id.upper()
    if TRUTH.get(cid, {}).get("split") == "test":
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
