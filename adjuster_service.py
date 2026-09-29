"""
Phase 5: the logic behind the adjuster app (app.py). No Streamlit here, so it can be tested.

It drives the SAME LangGraph workflow as run_graph.py, with the same saved checkpoints
(outputs/checkpoints.sqlite). A claim paused in the terminal shows up in the app, and the
other way round. It never changes agent code, so the tested system stays exactly the same.

Only dev claims are offered: the held-out test claims stay untouched.
The adjuster never sees the "expected" answers (ground truth); the app is for deciding,
not for grading.
"""

import json
import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from claim_graph import build_graph

CLAIMS_DIR = Path("data/claims")
CHECKPOINTS = Path("outputs/checkpoints.sqlite")
TRUTH = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
DEV_IDS = sorted(c for c, t in TRUTH.items() if t["split"] == "dev")
DECISIONS = ["approve", "investigate", "deny"]

STEP_LABELS = {
    "intake": "Reading the claim form, estimate and photo",
    "request_info": "Documents incomplete: waiting for a human",
    "coverage": "Coverage agent: policy rules and policy wording",
    "evidence": "Evidence agent: repair cost and claimant's story",
    "fraud": "Fraud agent: reused photos, invoices, timing",
    "decide": "Decision agent: recommendation and note",
    "critic": "Critic: checking the note for invented facts",
    "human_review": "Adjuster decision",
    "finalize": "Claim closed",
}


# ---------------------------------------------------------------------------
# graph + state
# ---------------------------------------------------------------------------
def open_graph(path=None):
    # CLAIM_CHECKPOINTS lets tests use a temporary database instead of your real one
    path = Path(path or os.getenv("CLAIM_CHECKPOINTS") or CHECKPOINTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)  # parallel nodes use threads
    return build_graph(SqliteSaver(conn))


def config_for(cid, use_llm=False):
    return {"configurable": {"thread_id": cid, "use_llm": use_llm}}


def _check(cid):
    if cid not in DEV_IDS:
        raise ValueError(f"{cid} is not a dev claim. Test claims stay untouched.")


def claim_state(graph, cid):
    """{'status', 'values', 'pause'} for one claim.
    status: not_started | waiting_adjuster | waiting_documents | decided | closed | in_progress"""
    snap = graph.get_state(config_for(cid))
    values = snap.values or {}
    pause = None
    for task in snap.tasks:
        if task.interrupts:
            pause = task.interrupts[0].value
    if not values:
        status = "not_started"
    elif pause and pause.get("type") == "adjuster_review":
        status = "waiting_adjuster"
    elif pause:
        status = "waiting_documents"
    elif values.get("status") == "decided":
        status = "decided"
    elif values.get("status") == "closed_incomplete":
        status = "closed"
    else:
        status = "in_progress"
    return {"status": status, "values": values, "pause": pause}


def saved_form(cid):
    """Claim form fields from the saved intake (so a claim can be listed before it is processed)."""
    try:
        from data_access import load_intake
        return load_intake(cid).claim_form.model_dump()
    except FileNotFoundError:
        return {}


def list_claims(graph):
    """One row per dev claim for the queue table (no ground truth)."""
    rows = []
    for cid in DEV_IDS:
        st = claim_state(graph, cid)
        v = st["values"]
        form = (v.get("intake") or {}).get("claim_form") or saved_form(cid)
        rows.append({
            "claim": cid,
            "claimant": form.get("claimant_name") or "",
            "amount_pkr": form.get("amount_claimed"),
            "status": st["status"],
            "recommendation": v.get("recommendation") or "",
            "final_decision": v.get("final_decision") or "",
            "overridden": bool((v.get("human") or {}).get("overridden")),
        })
    return rows


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
def process(graph, cid, use_llm=False):
    """Run a claim until it pauses. Yields one event per finished step, live:
    {"node", "label", "summary"}; the last event is {"node": "__pause__", "pause": payload}."""
    _check(cid)
    st = claim_state(graph, cid)
    if st["status"] != "not_started":
        raise ValueError(f"{cid} was already processed (status: {st['status']}). Use 'start over' first.")
    yield from _stream(graph, {"claim_id": cid}, config_for(cid, use_llm))


def _stream(graph, inp, cfg):
    for update in graph.stream(inp, cfg, stream_mode="updates"):
        for node, change in update.items():
            if node == "__interrupt__":
                yield {"node": "__pause__", "pause": change[0].value}
                continue
            summary = (change or {}).get("audit", [{}])[-1].get("summary", "") if change else ""
            yield {"node": node, "label": STEP_LABELS.get(node, node), "summary": summary}


def decide(graph, cid, action, note="", reviewer="adjuster"):
    """The adjuster's answer to a paused claim. action: accept | approve | investigate | deny
    (for incomplete documents: continue | close). Returns the new claim state."""
    _check(cid)
    st = claim_state(graph, cid)
    allowed = {"waiting_adjuster": ["accept"] + DECISIONS, "waiting_documents": ["continue", "close"]}
    if action not in allowed.get(st["status"], []):
        raise ValueError(f"{cid} is '{st['status']}': '{action}' is not possible now.")
    if not (reviewer or "").strip():
        raise ValueError("Enter the reviewer's name: every decision is signed.")
    cfg = config_for(cid, use_llm=False)  # the remaining steps do not call the AI
    for _ in _stream(graph, Command(resume={"action": action, "note": note.strip(),
                                            "reviewer": reviewer.strip()}), cfg):
        pass
    return claim_state(graph, cid)


def start_over(graph, cid):
    """Forget a claim's saved run (its audit trail too), so it can be processed again."""
    _check(cid)
    graph.checkpointer.delete_thread(cid)


# ---------------------------------------------------------------------------
# what the adjuster sees
# ---------------------------------------------------------------------------
def details(values):
    """Everything the review screen shows, in plain Python types."""
    intake = values.get("intake") or {}
    form, photo = intake.get("claim_form") or {}, intake.get("photo") or {}
    cov, ev, fr = values.get("coverage") or {}, values.get("evidence") or {}, values.get("fraud") or {}
    flags = [{"from": "evidence", **f} for f in ev.get("flags", [])] + \
            [{"from": "fraud", **f} for f in fr.get("signals", [])]
    order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda f: order.get(f["severity"], 3))
    cid = values.get("claim_id", "")
    return {
        "claim_id": cid,
        "photo_path": str(CLAIMS_DIR / cid / "photo_1.jpg") if cid else None,
        "form": form,
        "photo": photo,
        "coverage": cov,
        "quotes": [c for c in cov.get("citations", []) if c.get("verified")],
        "evidence": ev,
        "flags": flags,
        "writeup": values.get("writeup") or {},
        "recommendation": values.get("recommendation"),
        "rule_recommendation": values.get("rule_recommendation"),
        "human": values.get("human") or {},
        "final_decision": values.get("final_decision"),
        "audit": values.get("audit") or [],
        "intake_issues": intake.get("issues", []),
        "critic": values.get("critic") or {},
        "raw": {"intake": intake, "coverage": cov, "evidence": ev, "fraud": fr,
                "decision": values.get("writeup") or {}, "critic": values.get("critic") or {}},
    }


# How each agent works, shown next to its raw output in the app
AGENT_HOW = {
    "intake": "AI (gpt-5.4-mini) read the claim form, the repair estimate and the photo; plain code "
              "then checked the fields (dates, amounts, matching names).",
    "coverage": "Plain code checked the policy is on record, active and comprehensive. Then search "
                "(keywords + embeddings) found the relevant clauses in the policy PDF, the AI (nano) judged "
                "them, and plain code verified every quote word for word.",
    "evidence": "Plain code compared the amount with the typical repair cost for the damage in the photo. "
                "The AI (nano) read the claimant's story and checked it for exaggeration.",
    "fraud": "Plain code only: photo fingerprints against earlier claims, repeated invoice numbers, "
             "a policy started days before the incident, late reports, repeat claimants.",
    "decision": "Plain rules chose the recommendation; the AI (nano) wrote the note for the adjuster.",
    "critic": "Plain code only: rejects notes with invented flags, unverified quotes, numbers not in "
              "the findings, or 'nothing found' when something was found.",
}


# ---------------------------------------------------------------------------
# results page (evaluation numbers, read from saved files)
# ---------------------------------------------------------------------------
RESULT_FILES = {"test": Path("outputs/final_test.json"), "test2": Path("outputs/final_test2.json"),
                "test3": Path("outputs/final_test3.json")}


def results():
    """Saved held-out results. Test 2 and 3 are combined only if they tested the same code."""
    out = {}
    for name, path in RESULT_FILES.items():
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("complete"):
                out[name] = d
    combined = None
    if "test2" in out and "test3" in out and out["test2"]["fingerprint"] == out["test3"]["fingerprint"]:
        rows = out["test2"]["rows"] + out["test3"]["rows"]
        combined = _summary(rows)
    decisions = []
    log = Path("outputs/policy_decisions.jsonl")
    if log.exists():
        decisions = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"tests": out, "combined": combined, "decisions": decisions}


def _summary(rows):
    n = len(rows)
    ok = sum(r["got"] == r["expected"] for r in rows)
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], [0, 0])
        by[r["scenario"]][0] += r["got"] == r["expected"]
        by[r["scenario"]][1] += 1
    return {"claims": n, "correct": ok, "accuracy": ok / n if n else None,
            "wrong_approvals": sum(r["got"] == "approve" and r["expected"] != "approve" for r in rows),
            "wrong_denials": sum(r["got"] == "deny" and r["expected"] != "deny" for r in rows),
            "honest_held_up": sum(r["scenario"] == "honest" and r["got"] != "approve" for r in rows),
            "by_scenario": by}
