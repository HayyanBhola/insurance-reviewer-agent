"""Phase 4 graph tests with a stand-in AI writer (no API calls, free). Run: pytest -q"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import claim_graph  # noqa: E402
import decision_agent  # noqa: E402
from decision_agent import DecisionWriteup, critique, template_writeup  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

TRUTH = json.loads((ROOT / "data/ground_truth.json").read_text(encoding="utf-8"))
HAVE_DATA = (ROOT / "outputs/intake").exists() and any((ROOT / "outputs/intake").iterdir())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="needs outputs/intake from run_intake.py")


def _first(scenario):
    return next(c for c, t in TRUTH.items() if t["split"] == "dev" and t["scenario"] == scenario
                and (ROOT / f"outputs/intake/{c}.json").exists())


def _run(cid, use_llm, answer=None, monkeypatch=None):
    """Run a claim through the graph until it pauses, then answer the pause."""
    graph = claim_graph.build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": cid, "use_llm": use_llm}}
    out = graph.invoke({"claim_id": cid}, cfg)
    assert "__interrupt__" in out
    graph.invoke(Command(resume=answer or {"action": "accept"}), cfg)
    return graph.get_state(cfg).values


def good(findings, rec, feedback=None):
    return template_writeup(findings, rec)


def bad(findings, rec, feedback=None):
    return DecisionWriteup(summary="Recommend it. Claimed PKR 987,654 on 2031-01-01.",
                           justification="Invented details.", flag_codes_cited=["made_up_flag"],
                           policy_quotes_used=["The policy covers everything always."],
                           suggest_escalation=False, escalation_reason="")


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.chdir(ROOT)
    # These tests exercise the decision writer and critic with a stand-in AI. Keep the
    # coverage and evidence agents in rules mode so no real API is ever called.
    real_cov, real_ev = claim_graph.check_coverage, claim_graph.check_evidence
    monkeypatch.setattr(claim_graph, "check_coverage", lambda i, ix, use_llm=True: real_cov(i, ix, use_llm=False))
    monkeypatch.setattr(claim_graph, "check_evidence", lambda i, use_llm=True: real_ev(i, use_llm=False))
    # and never let a test reach a real LLM through the decision writer by accident
    monkeypatch.setattr(claim_graph, "llm_writeup", good)


# ---------------------------------------------------------------------------
# critic (pure functions)
# ---------------------------------------------------------------------------
FINDINGS = {
    "coverage": {"status": "covered", "reasons": ["Policy active on 2026-05-01."], "citations": [
        {"quote": "damage to tyres and tubes unless the vehicle is damaged", "verified": True}],
        "coverage_type": "comprehensive", "deductible": 5000.0, "sum_insured": 1500000.0},
    "evidence": {"flags": [], "photo_summary": "moderate dent", "claimed_amount": 31800.0,
                 "typical_range_pkr": [30000.0, 80000.0], "cost_ratio": 0.4, "cost_status": "within",
                 "description_note": "Matches."},
    "fraud": {"signals": [{"code": "new_policy", "severity": "medium", "message": "Only 3 days."}]},
}


def test_critic_rejects_invented_facts():
    report = critique(bad(FINDINGS, "investigate"), FINDINGS, "investigate")
    text = " ".join(report.problems)
    assert not report.passed
    assert "made_up_flag" in text and "987654" in text and "quote" in text


def test_critic_accepts_amounts_written_with_commas():
    w = DecisionWriteup(summary="Investigate: policy started recently.",
                        justification="Claimed PKR 31,800, within PKR 30,000-80,000.",
                        flag_codes_cited=["new_policy"], policy_quotes_used=[],
                        suggest_escalation=False, escalation_reason="")
    assert critique(w, FINDINGS, "investigate").passed


def test_critic_rejects_saying_nothing_was_found_when_something_was():
    w = DecisionWriteup(summary="Approve: covered and no fraud signals were found.",
                        justification="Everything is fine.", flag_codes_cited=[], policy_quotes_used=[],
                        suggest_escalation=False, escalation_reason="")
    report = critique(w, FINDINGS, "approve")
    assert not report.passed and "new_policy" in " ".join(report.problems)


def test_template_mentions_low_severity_signals():
    low = {**FINDINGS, "fraud": {"signals": [
        {"code": "repeat_claimant", "severity": "low", "message": "Same claimant filed C010."}]}}
    w = template_writeup(low, "approve")
    assert "C010" in w.justification and "repeat_claimant" in w.flag_codes_cited
    assert "no risk signals" not in w.summary
    assert critique(w, low, "approve").passed


def test_template_always_passes_critic():
    for rec in ("approve", "investigate", "deny"):
        assert critique(template_writeup(FINDINGS, rec), FINDINGS, rec).passed


# ---------------------------------------------------------------------------
# full graph
# ---------------------------------------------------------------------------
@needs_data
def test_parallel_agents_all_write_to_the_audit_trail():
    s = _run(_first("honest"), use_llm=False)
    nodes = [a["node"] for a in s["audit"]]
    assert {"coverage", "evidence", "fraud"} <= set(nodes)       # reducer kept all three
    assert nodes[-1] == "finalize" and s["status"] == "decided"


@needs_data
def test_adjuster_override_is_recorded():
    s = _run(_first("honest"), use_llm=False, answer={"action": "deny", "note": "forged", "reviewer": "Ali"})
    assert s["final_decision"] == "deny" and s["human"]["overridden"] and s["human"]["note"] == "forged"


@needs_data
def test_critic_loop_rewrites_then_passes(monkeypatch):
    attempts = []

    def writer(findings, rec, feedback=None):
        attempts.append(feedback)
        return bad(findings, rec) if len(attempts) == 1 else good(findings, rec)

    monkeypatch.setattr(claim_graph, "llm_writeup", writer)
    s = _run(_first("honest"), use_llm=True)
    assert len(attempts) == 2 and attempts[0] is None and attempts[1]  # 2nd try got the feedback
    assert s["critic"]["passed"] and not s["used_template"]


@needs_data
def test_bad_writer_falls_back_to_safe_template(monkeypatch):
    monkeypatch.setattr(claim_graph, "llm_writeup", bad)
    s = _run(_first("honest"), use_llm=True)
    assert s["used_template"] and s["critic"]["passed"]
    assert s["revisions"] == decision_agent.MAX_REVISIONS + 2  # 3 AI attempts + 1 template


@needs_data
def test_ai_can_only_escalate_never_approve(monkeypatch):
    def escalate(findings, rec, feedback=None):
        w = good(findings, rec)
        return w.model_copy(update={"suggest_escalation": True, "escalation_reason": "odd timing"})

    monkeypatch.setattr(claim_graph, "llm_writeup", escalate)
    assert _run(_first("honest"), use_llm=True)["final_decision"] == "investigate"
    lapsed = _run(_first("lapsed_policy"), use_llm=True)
    assert lapsed["recommendation"] == "deny"  # escalation can't change a deny


@needs_data
def test_incomplete_documents_pause_and_can_be_closed(monkeypatch):
    from data_access import load_intake
    cid = _first("honest")

    def incomplete(c):
        r = load_intake(c)
        return r.model_copy(update={"ready_for_review": False, "missing_fields": ["claim_form.policy_number"]})

    monkeypatch.setattr(claim_graph, "load_intake", incomplete)
    graph = claim_graph.build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "x", "use_llm": False}}
    out = graph.invoke({"claim_id": cid}, cfg)
    assert out["__interrupt__"][0].value["type"] == "missing_information"
    graph.invoke(Command(resume={"action": "close"}), cfg)
    s = graph.get_state(cfg).values
    assert s["status"] == "closed_incomplete" and "coverage" not in s
