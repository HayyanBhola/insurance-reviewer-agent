"""Story check + coverage prompt + run comparison. Stand-in AI, no API calls. Run: pytest -q"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRUTH = json.loads((ROOT / "data/ground_truth.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("AI_CHECKS", "v2")  # tests say which version they test


def _intake(scenario="honest"):
    from data_access import load_intake
    cid = next(c for c, t in TRUTH.items() if t["split"] == "dev" and t["scenario"] == scenario
               and (ROOT / f"outputs/intake/{c}.json").exists())
    return load_intake(cid)


def _story(monkeypatch, **answer):
    import evidence_agent
    from findings import DescriptionCheck
    base = {"exaggerates": False, "claimed_damage_summary": "dent", "differences": "none",
            "explanation": "ok"}
    monkeypatch.setattr(evidence_agent, "structured_call",
                        lambda schema, messages, **kw: DescriptionCheck(**{**base, **answer}))
    return evidence_agent.check_evidence(_intake(), use_llm=True)


def test_exaggeration_is_a_high_flag(monkeypatch):
    f = _story(monkeypatch, exaggerates=True, explanation="Says destroyed; photo shows a scratch.")
    assert [(x.code, x.severity) for x in f.flags if x.code.startswith("description")] == \
        [("description_exaggerates_damage", "high")]
    assert f.description_consistent is False


def test_side_or_wording_difference_is_only_a_low_note(monkeypatch):
    f = _story(monkeypatch, differences="Claimant says front tyre; photo reading says rear tyre.")
    flags = [x for x in f.flags if x.code.startswith("description")]
    assert [(x.code, x.severity) for x in flags] == [("description_differs_from_photo", "low")]
    assert f.description_consistent is True


@pytest.mark.parametrize("nothing", ["none", "None.", "", "n/a", "No differences.", "No significant differences",
                                     "no discrepancies noted.", "N/A"])
def test_no_difference_no_flag(monkeypatch, nothing):
    f = _story(monkeypatch, differences=nothing)
    assert not [x for x in f.flags if x.code.startswith("description")]


def test_low_note_does_not_change_the_recommendation(monkeypatch):
    from decision_agent import recommend
    f = _story(monkeypatch, differences="front vs rear")
    findings = {"coverage": {"status": "covered", "reasons": ["ok"], "citations": []},
                "evidence": f.model_dump(), "fraud": {"signals": []}}
    assert recommend(findings) == "approve"


def test_coverage_prompt_puts_the_burden_on_exclusions():
    from coverage_agent import COVERAGE_SYSTEM
    text = " ".join(COVERAGE_SYSTEM.lower().split())
    assert "unless an exclusion applies" in text
    assert "a missing mention of a part is not a reason for needs_review" in text


def test_keep_rule():
    from run_graph import keep_rule

    def rows(**got):
        return {c: {"scenario": s, "expected": e, "got": got.get(c, e)} for c, (s, e) in {
            "A": ("honest", "approve"), "B": ("honest", "approve"), "F": ("exaggerated_damage", "investigate"),
            "T": ("excluded_tyre", "deny")}.items()}

    old = rows(A="investigate", B="investigate")
    assert keep_rule(old, rows(B="investigate"))[0]                      # honest improved, nothing broke
    assert not keep_rule(old, rows(A="investigate", B="investigate"))[0]  # no honest gain
    assert not keep_rule(old, rows(B="investigate", F="approve"))[0]     # fraud got through
    assert not keep_rule(old, rows(B="investigate", T="investigate"))[0]  # tyre got worse


def test_saved_baseline_is_complete():
    data = json.loads((ROOT / "outputs/graph_runs/llm_nano_v1.json").read_text(encoding="utf-8"))
    assert len(data["rows"]) == 48
    assert sum(r["got"] == r["expected"] for r in data["rows"].values()) == 32


@pytest.mark.parametrize("something", ["Front vs rear tyre.", "No mention of the dent in the description.",
                                       "Photo also shows a scratch on the door."])
def test_real_differences_are_kept(something):
    from evidence_agent import says_nothing
    assert not says_nothing(something)


def test_no_verdict_on_an_incomplete_run(tmp_path, monkeypatch):
    import run_graph
    monkeypatch.setattr(run_graph, "RUNS", tmp_path)
    full = json.loads((ROOT / "outputs/graph_runs/llm_nano_v1.json").read_text(encoding="utf-8"))
    (tmp_path / "old.json").write_text(json.dumps(full), encoding="utf-8")
    part = {**full, "rows": dict(list(full["rows"].items())[:40])}
    (tmp_path / "new.json").write_text(json.dumps(part), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_graph.compare_runs("old", "new")
    with pytest.raises(SystemExit):
        run_graph.compare_runs("old", "typo_name")


# ---------------------------------------------------------------------------
# v1 is kept, word for word, and still behaves as before
# ---------------------------------------------------------------------------
def test_v1_story_check_flags_any_mismatch_high(monkeypatch):
    import evidence_agent
    from findings import MatchCheck
    monkeypatch.setenv("AI_CHECKS", "v1")
    seen = {}

    def fake(schema, messages, **kw):
        seen["schema"], seen["question"] = schema, messages[1].content
        return MatchCheck(consistent=False, claimed_damage_summary="front tyre", explanation="front vs rear")

    monkeypatch.setattr(evidence_agent, "structured_call", fake)
    f = evidence_agent.check_evidence(_intake(), use_llm=True)
    assert seen["schema"] is MatchCheck and seen["question"].endswith("Do they match?")
    assert [(x.code, x.severity) for x in f.flags if x.code.startswith("description")] == \
        [("description_does_not_match_photo", "high")]


def test_coverage_prompt_follows_the_switch(monkeypatch):
    import coverage_agent
    from policy_index import PolicyIndex
    monkeypatch.setattr(PolicyIndex, "_embed_query", lambda self, q: None)  # keywords only: no API in tests
    seen = []
    monkeypatch.setattr(coverage_agent, "structured_call",
                        lambda schema, messages, **kw: seen.append(messages[0].content) or (_ for _ in ()).throw(StopIteration))
    for v, text in (("v1", coverage_agent.COVERAGE_SYSTEM_V1), ("v2", coverage_agent.COVERAGE_SYSTEM)):
        monkeypatch.setenv("AI_CHECKS", v)
        with pytest.raises((StopIteration, RuntimeError)):
            coverage_agent.check_coverage(_intake(), coverage_agent_index(), use_llm=True)
        assert seen[-1] == text


def coverage_agent_index():
    from policy_index import INDEX_FILE, PolicyIndex
    if not INDEX_FILE.exists():
        pytest.skip("needs the policy index")
    return PolicyIndex()


def test_unknown_checks_version_is_refused(monkeypatch):
    from llm import checks_version
    monkeypatch.setenv("AI_CHECKS", "v7")
    with pytest.raises(ValueError):
        checks_version()


def test_decision_was_recorded_before_the_test_set():
    d = json.loads((ROOT / "outputs/policy_decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "AI_CHECKS=v2" in d["decision"] and "No changes" in d["final_test_plan"]


# ---------------------------------------------------------------------------
# AI_ESCALATION and TYRE_REVIEW switches
# ---------------------------------------------------------------------------
def test_escalation_switch(monkeypatch):
    import claim_graph
    from decision_agent import template_writeup
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    cid = next(c for c, t in TRUTH.items() if t["split"] == "dev" and t["scenario"] == "honest"
               and (ROOT / f"outputs/intake/{c}.json").exists())
    monkeypatch.setattr(claim_graph, "check_coverage", lambda i, ix, use_llm=True:
                        __import__("coverage_agent").check_coverage(i, ix, use_llm=False))
    monkeypatch.setattr(claim_graph, "check_evidence", lambda i, use_llm=True:
                        __import__("evidence_agent").check_evidence(i, use_llm=False))
    monkeypatch.setattr(claim_graph, "llm_writeup", lambda f, rec, feedback=None: template_writeup(f, rec).model_copy(
        update={"suggest_escalation": True, "escalation_reason": "odd timing"}))
    results = {}
    for mode in ("on", "off"):
        monkeypatch.setenv("AI_ESCALATION", mode)
        g = claim_graph.build_graph(InMemorySaver())
        cfg = {"configurable": {"thread_id": mode, "use_llm": True}}
        g.invoke({"claim_id": cid}, cfg)
        g.invoke(Command(resume={"action": "accept"}), cfg)
        results[mode] = g.get_state(cfg).values
    assert results["on"]["final_decision"] == "investigate"
    assert results["off"]["final_decision"] == "approve"
    assert "switched off" in results["off"]["audit"][-4]["summary"] or \
        any("switched off" in a["summary"] for a in results["off"]["audit"])


@pytest.mark.parametrize("text,expected", [
    ("The front tire burst while driving and went completely flat.", True),
    ("My tyre got a puncture on the motorway.", True),
    ("A truck hit my car. The front tire is destroyed and both headlights smashed.", False),
    ("Another car hit my vehicle, causing a large dent on the front fender.", False),
    ("", False)])
def test_tyre_story_detection(text, expected):
    from coverage_agent import story_is_tyre_only
    assert story_is_tyre_only(text) is expected


def test_tyre_review_sends_tyre_story_with_body_damage_to_a_human(monkeypatch):
    from coverage_agent import check_coverage
    from schemas import PhotoAssessment
    cid = next(c for c, t in TRUTH.items() if t["split"] == "dev" and t["scenario"] == "excluded_tyre"
               and (ROOT / f"outputs/intake/{c}.json").exists())
    r = _intake("excluded_tyre")
    tyre_story = r.claim_form.model_copy(update={"description": "The front tire burst while driving."})

    def photo(types):
        return PhotoAssessment(is_vehicle_photo=True, damaged_part="front tire", damage_type="tire flat",
                               severity="moderate", description="", image_quality_ok=True,
                               all_damages=[{"part": "p", "damage_type": t, "severity": "moderate"} for t in types])

    both = r.model_copy(update={"claim_form": tyre_story, "photo": photo(["tire flat", "dent"])})
    only = r.model_copy(update={"claim_form": tyre_story, "photo": photo(["tire flat"])})
    monkeypatch.setenv("TYRE_REVIEW", "off")
    assert check_coverage(both, None, use_llm=False).status == "covered"       # original behaviour
    monkeypatch.setenv("TYRE_REVIEW", "on")
    assert check_coverage(both, None, use_llm=False).status == "needs_review"  # a human checks
    assert check_coverage(only, None, use_llm=False).status == "needs_review"  # tyre-only path unchanged
    assert cid


def test_money_out_rule():
    from run_graph import keep_rule_money_out

    def rows(**got):
        base = {"T": ("excluded_tyre", "deny"), "H1": ("honest", "approve"), "H2": ("honest", "approve"),
                "H3": ("honest", "approve"), "F": ("inflated_estimate", "investigate")}
        return {c: {"scenario": s, "expected": e, "got": got.get(c, e)} for c, (s, e) in base.items()}

    old = rows(T="approve")
    assert keep_rule_money_out(old, rows(T="investigate", H1="investigate", H2="investigate"))[0]
    assert not keep_rule_money_out(old, rows(T="investigate", H1="investigate", H2="investigate",
                                             H3="investigate"))[0]            # 3 more held up: too many
    assert not keep_rule_money_out(old, rows(T="approve"))[0]                 # no fewer wrong approvals
    assert not keep_rule_money_out(old, rows(T="investigate", F="approve"))[0]  # fraud got through
