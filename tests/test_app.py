"""Phase 5 app: logic (adjuster_service) and screens (app.py via Streamlit's AppTest).
Rules-only mode and a temporary checkpoint database: no API calls, your saved claims untouched."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAVE_DATA = (ROOT / "outputs/intake").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="needs outputs/intake")


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("CLAIM_CHECKPOINTS", str(tmp_path / "checkpoints.sqlite"))
    import streamlit as st
    st.cache_resource.clear()   # each test gets its own fresh database connection


def _honest():
    t = json.loads((ROOT / "data/ground_truth.json").read_text(encoding="utf-8"))
    return next(c for c, v in sorted(t.items()) if v["split"] == "dev" and v["scenario"] == "honest"
                and (ROOT / f"outputs/intake/{c}.json").exists())


# ---------------------------------------------------------------------------
# logic
# ---------------------------------------------------------------------------
@needs_data
def test_process_streams_every_step_then_pauses():
    import adjuster_service as s
    g = s.open_graph()
    events = list(s.process(g, _honest(), use_llm=False))
    nodes = [e["node"] for e in events]
    assert {"intake", "coverage", "evidence", "fraud", "decide", "critic"} <= set(nodes)
    assert nodes[-1] == "__pause__" and events[-1]["pause"]["type"] == "adjuster_review"


@needs_data
def test_decision_is_signed_recorded_and_final():
    import adjuster_service as s
    g, cid = s.open_graph(), _honest()
    list(s.process(g, cid))
    with pytest.raises(ValueError):
        s.decide(g, cid, "deny", reviewer="  ")                 # unsigned: refused
    st = s.decide(g, cid, "deny", note="edited invoice", reviewer="Shamim")
    assert st["status"] == "decided" and st["values"]["human"]["overridden"]
    with pytest.raises(ValueError):
        s.decide(g, cid, "approve", reviewer="Shamim")           # already decided
    with pytest.raises(ValueError):
        list(s.process(g, cid))                                   # already processed
    s.start_over(g, cid)
    assert s.claim_state(g, cid)["status"] == "not_started"


def test_test_claims_are_refused():
    import adjuster_service as s
    t = json.loads((ROOT / "data/ground_truth.json").read_text(encoding="utf-8"))
    test_ids = [c for c, v in t.items() if v["split"] != "dev"]
    assert test_ids and not set(test_ids) & set(s.DEV_IDS)
    g = s.open_graph()
    with pytest.raises(ValueError):
        list(s.process(g, test_ids[0]))


@needs_data
def test_queue_rows_never_show_the_answer_key():
    import adjuster_service as s
    row = s.list_claims(s.open_graph())[0]
    assert not {"scenario", "expected", "expected_decision"} & set(row)


# ---------------------------------------------------------------------------
# screens
# ---------------------------------------------------------------------------
def _app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.toggle[0].set_value(False).run()   # rules only: free, no API
    assert not at.exception, at.exception
    return at


def test_app_starts_with_four_tabs():
    at = _app()
    assert [t.label for t in at.tabs][:4] == ["📥 Review queue", "▶️ Process a claim", "📜 Audit trail", "📊 Results"]


@needs_data
def test_full_flow_process_review_override_audit():
    at = _app()
    process_button = next(b for b in at.button if b.label.startswith("Process C"))
    cid = process_button.label.split()[-1]
    process_button.click().run()
    assert not at.exception, at.exception
    # no manual refresh: the processed claim is in the queue straight away
    assert any("Waiting for you" == m.label and m.value == "1" for m in at.metric)
    assert any(cid in o for o in at.selectbox[0].options)      # queue dropdown lists it
    import adjuster_service as svc
    rec = svc.claim_state(svc.open_graph(), cid)["values"]["recommendation"]
    other = "approve" if rec != "approve" else "deny"            # really disagree with the AI
    at.radio[0].set_value(other)
    at.text_area[0].input("Photo looks staged")
    next(b for b in at.button if b.label == "Submit decision").click().run()
    assert not at.exception, at.exception
    assert any(f"{cid}: {other.upper()} (OVERRIDE)" in s.value for s in at.success)
    assert any("Decided" == m.label and m.value == "1" for m in at.metric)
    assert any("overrode" in w.value for w in at.warning)      # audit tab shows the override


def test_results_tab_reads_saved_results(tmp_path, monkeypatch):
    import adjuster_service as s
    row = {"scenario": "honest", "expected": "approve", "got": "approve", "photo_type_ok": True,
           "photo_within_one": True}
    bad = {**row, "scenario": "inflated_estimate", "expected": "investigate", "got": "approve"}
    files = {}
    for name, rows, fp in (("test", [row, bad], "a"), ("test2", [row, row], "b"), ("test3", [row, bad], "b")):
        p = tmp_path / f"{name}.json"
        summ = {"claims": len(rows), "correct": sum(r["got"] == r["expected"] for r in rows),
                "wrong_approvals": sum(r is bad for r in rows)}
        summ["accuracy"] = summ["correct"] / summ["claims"]
        p.write_text(json.dumps({"complete": True, "fingerprint": fp, "rows": rows, "summary": summ}), encoding="utf-8")
        files[name] = p
    monkeypatch.setattr(s, "RESULT_FILES", files)
    r = s.results()
    assert r["combined"]["claims"] == 4 and r["combined"]["wrong_approvals"] == 1
    files["test3"].write_text(json.dumps({**json.loads(files["test3"].read_text()), "fingerprint": "zzz"}), encoding="utf-8")
    assert s.results()["combined"] is None                     # different systems are never combined


@needs_data
def test_accepting_the_recommendation_is_not_an_override():
    at = _app()
    next(b for b in at.button if b.label.startswith("Process C")).click().run()
    at.run()
    next(b for b in at.button if b.label == "Submit decision").click().run()   # default choice: accept
    assert any("(accepted)" in s.value for s in at.success)
    assert not at.warning


@needs_data
def test_review_shows_every_agents_raw_output():
    at = _app()
    next(b for b in at.button if b.label.startswith("Process C")).click().run()
    at.run()
    assert not at.exception, at.exception
    assert any(e.label.startswith("🔎 What each agent returned") for e in at.expander)
    labels = [t.label for t in at.tabs]
    for name in ("Intake", "Coverage", "Evidence", "Fraud", "Decision", "Critic"):
        assert name in labels
    assert len(at.json) >= 6
