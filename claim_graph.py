"""
Phase 4: the claims workflow as a LangGraph StateGraph.

  START -> intake --(documents missing?)--> request_info (pause) --> END / continue
             |
             +--> coverage --+
             +--> evidence --+--> decide <--+        coverage, evidence and fraud run in PARALLEL;
             +--> fraud -----+       |      |        decide waits for all three (a join)
                                   critic --+ fails (max 2 rewrites, then safe template)
                                     |
                               human_review (PAUSE: adjuster accepts or overrides)
                                     |
                                  finalize -> END

Concepts shown: StateGraph, typed state, parallel fan-out and join, a reducer for the
audit trail, conditional edges, a bounded reflection loop, interrupt() for
human-in-the-loop, and a checkpointer so a claim can pause for days and resume.
"""

import operator
from datetime import datetime
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from coverage_agent import check_coverage
from data_access import build_claims_history, load_intake
from decision_agent import (MAX_REVISIONS, critique, llm_writeup, recommend,
                            template_writeup)
from llm import ai_escalation
from evidence_agent import check_evidence
from fraud_agent import check_fraud
from policy_index import INDEX_FILE, PolicyIndex
from schemas import IntakeResult


# ---------------------------------------------------------------------------
# State: everything the graph knows about one claim. Plain dicts, so the
# checkpointer can save it to disk and a claim can resume later.
# ---------------------------------------------------------------------------
class ClaimState(TypedDict, total=False):
    claim_id: str
    intake: dict
    coverage: dict
    evidence: dict
    fraud: dict
    rule_recommendation: str   # what the rules said
    recommendation: str        # after a possible AI escalation (approve -> investigate only)
    writeup: dict
    critic: dict
    revisions: int
    used_template: bool
    human: dict
    final_decision: str
    status: str
    # reducer: every node APPENDS its audit entry; parallel nodes can't overwrite each other
    audit: Annotated[list[dict], operator.add]


def _audit(node, summary):
    return [{"node": node, "time": datetime.now().isoformat(timespec="seconds"), "summary": summary}]


def _use_llm(config):
    return bool((config or {}).get("configurable", {}).get("use_llm", False))


_shared = {}


def _index():
    if "index" not in _shared:
        _shared["index"] = PolicyIndex() if INDEX_FILE.exists() else None
    return _shared["index"]


def _history():
    if "history" not in _shared:
        _shared["history"] = build_claims_history()
    return _shared["history"]


def _intake(state) -> IntakeResult:
    return IntakeResult.model_validate(state["intake"])


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def intake_node(state: ClaimState, config=None):
    """Reuse the saved Phase 2 result (free). Only call the AI if it doesn't exist yet."""
    cid = state["claim_id"]
    try:
        result = load_intake(cid)
        how = "loaded saved intake"
    except FileNotFoundError:
        if not _use_llm(config):
            raise
        from intake import run_intake
        result = run_intake(cid)
        how = "ran intake agent"
    ready = result.ready_for_review
    return {"intake": result.model_dump(), "revisions": 0,
            "audit": _audit("intake", f"{how}; ready_for_review={ready}; "
                                      f"{len(result.issues)} issue(s), {len(result.missing_fields)} missing")}


def request_info_node(state: ClaimState):
    """Documents are incomplete: pause and ask a human what to do."""
    it = state["intake"]
    answer = interrupt({
        "type": "missing_information",
        "claim_id": state["claim_id"],
        "missing_fields": it["missing_fields"],
        "issues": [i["message"] for i in it["issues"] if i["severity"] == "error"],
        "options": ["continue", "close"],
    })
    action = (answer or {}).get("action", "close")
    return {"status": "incomplete_continued" if action == "continue" else "closed_incomplete",
            "audit": _audit("request_info", f"human chose '{action}'")}


def coverage_node(state: ClaimState, config=None):
    f = check_coverage(_intake(state), _index(), use_llm=_use_llm(config))
    return {"coverage": f.model_dump(),
            "audit": _audit("coverage", f"{f.status} ({f.method}), {len(f.citations)} citation(s)")}


def evidence_node(state: ClaimState, config=None):
    f = check_evidence(_intake(state), use_llm=_use_llm(config))
    return {"evidence": f.model_dump(),
            "audit": _audit("evidence", f"cost {f.cost_status}, story " + {
                None: "not checked", True: "ok", False: "EXAGGERATED"}[f.description_consistent])}


def fraud_node(state: ClaimState):
    f = check_fraud(state["claim_id"], _intake(state), _history())
    return {"fraud": f.model_dump(),
            "audit": _audit("fraud", f"risk {f.risk}, {len(f.signals)} signal(s)")}


def decide_node(state: ClaimState, config=None):
    findings = {k: state[k] for k in ("coverage", "evidence", "fraud")}
    rule_rec = rec = recommend(findings)
    revisions = state.get("revisions", 0)
    feedback = (state.get("critic") or {}).get("problems")

    if _use_llm(config) and revisions <= MAX_REVISIONS:
        w = llm_writeup(findings, rec, feedback=feedback if revisions else None)
        used_template = False
        if w.suggest_escalation and rec == "approve":
            if ai_escalation():
                rec = "investigate"  # the AI may only make the outcome MORE cautious
            else:  # AI_ESCALATION=off: the rules decide; the AI's concern stays visible in the audit
                note_off = f"AI suggested escalation (switched off): {w.escalation_reason}"
                w = w.model_copy(update={"suggest_escalation": False, "escalation_reason": ""})
    else:
        w = template_writeup(findings, rec)
        used_template = True

    note = "template (rules-only)" if used_template else f"AI write-up, attempt {revisions + 1}"
    if not used_template and "note_off" in locals():
        note += f"; {note_off}"
    if rec != rule_rec:
        note += f"; AI escalated {rule_rec} -> {rec}: {w.escalation_reason}"
    return {"rule_recommendation": rule_rec, "recommendation": rec,
            "writeup": w.model_dump(), "used_template": used_template,
            "revisions": revisions + 1,
            "audit": _audit("decide", f"recommend {rec}; {note}")}


def critic_node(state: ClaimState):
    from decision_agent import DecisionWriteup
    findings = {k: state[k] for k in ("coverage", "evidence", "fraud")}
    # judge the note against what the RULES recommended (an escalation is allowed on approve)
    report = critique(DecisionWriteup.model_validate(state["writeup"]), findings,
                      state["rule_recommendation"])
    return {"critic": report.model_dump(),
            "audit": _audit("critic", "passed" if report.passed else f"failed: {report.problems[:2]}")}


def human_review_node(state: ClaimState):
    """PAUSE here. The graph saves its state and waits for the adjuster's decision."""
    answer = interrupt({
        "type": "adjuster_review",
        "claim_id": state["claim_id"],
        "recommendation": state["recommendation"],
        "escalated_by_ai": state["recommendation"] != state["rule_recommendation"],
        "escalation_reason": state["writeup"].get("escalation_reason", ""),
        "summary": state["writeup"]["summary"],
        "justification": state["writeup"]["justification"],
        "options": ["accept", "approve", "deny", "investigate"],
    }) or {}
    action = answer.get("action", "accept")
    final = state["recommendation"] if action == "accept" else action
    human = {"action": action, "final_decision": final, "note": answer.get("note", ""),
             "reviewer": answer.get("reviewer", "adjuster"),
             "overridden": final != state["recommendation"]}
    return {"human": human, "final_decision": final,
            "audit": _audit("human_review", f"{human['reviewer']} -> {final}"
                                            + (" (OVERRIDE)" if human["overridden"] else ""))}


def finalize_node(state: ClaimState):
    return {"status": "decided", "audit": _audit("finalize", f"final decision: {state['final_decision']}")}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
SPECIALISTS = ["coverage", "evidence", "fraud"]


def route_after_intake(state: ClaimState):
    return SPECIALISTS if state["intake"]["ready_for_review"] else "request_info"


def route_after_request_info(state: ClaimState):
    return SPECIALISTS if state["status"] == "incomplete_continued" else END


def route_after_critic(state: ClaimState):
    return "human_review" if state["critic"]["passed"] else "decide"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    g = StateGraph(ClaimState)
    g.add_node("intake", intake_node)
    g.add_node("request_info", request_info_node)
    g.add_node("coverage", coverage_node)
    g.add_node("evidence", evidence_node)
    g.add_node("fraud", fraud_node)
    g.add_node("decide", decide_node)
    g.add_node("critic", critic_node)
    g.add_node("human_review", human_review_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, SPECIALISTS + ["request_info"])
    g.add_conditional_edges("request_info", route_after_request_info, SPECIALISTS + [END])
    g.add_edge(SPECIALISTS, "decide")  # join: waits for all three parallel agents
    g.add_edge("decide", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["human_review", "decide"])
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
