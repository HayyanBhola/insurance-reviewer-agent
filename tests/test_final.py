"""Final test safety: settings check, one-time run, metrics. No API calls. Run: pytest -q"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_there_are_ten_untouched_test_claims():
    import final_test
    assert len(final_test.TEST_IDS) == 10
    assert all(final_test.TRUTH[c]["split"] == "test" for c in final_test.TEST_IDS)


def test_preflight_catches_wrong_settings(monkeypatch):
    import final_test
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for k, v in final_test.REQUIRED.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("AI_CHECKS", "v1")
    problems = final_test.preflight()
    assert any("AI_CHECKS" in p for p in problems)


def test_a_finished_final_test_is_never_run_again(monkeypatch, tmp_path, capsys):
    import final_test
    done = {"time": "t", "fingerprint": "x", "settings": {}, "complete": True, "rows": [],
            "summary": {"claims": 1, "correct": 1, "accuracy": 1.0, "wrong_approvals": 0, "wrong_denials": 0,
                        "honest_held_up": 0, "photo_type_correct": 1, "photo_severity_within_one": 1},
            "cost_usd": 0.1, "tokens_in": 1, "tokens_out": 1}
    path = tmp_path / "final_test.json"
    path.write_text(json.dumps(done), encoding="utf-8")
    monkeypatch.setattr(final_test, "RESULT", path)
    import claim_graph
    monkeypatch.setattr(claim_graph, "build_graph", lambda *a, **k: pytest.fail("must not run again"))
    final_test.run()
    assert "already been run" in capsys.readouterr().out


def test_summary_counts_the_costly_mistakes():
    from final_test import summarize
    rows = [
        {"scenario": "honest", "expected": "approve", "got": "investigate", "photo_type_ok": True, "photo_within_one": True},
        {"scenario": "excluded_tyre", "expected": "deny", "got": "approve", "photo_type_ok": True, "photo_within_one": True},
        {"scenario": "inflated_estimate", "expected": "investigate", "got": "investigate", "photo_type_ok": False, "photo_within_one": True},
        {"scenario": "honest", "expected": "approve", "got": "deny", "photo_type_ok": True, "photo_within_one": False},
    ]
    s = summarize(rows)
    assert (s["correct"], s["wrong_approvals"], s["wrong_denials"], s["honest_held_up"]) == (1, 1, 1, 2)


# ---------------------------------------------------------------------------
# second held-out set
# ---------------------------------------------------------------------------
def test_test2_requires_the_frozen_switches(monkeypatch):
    import final_test
    req = final_test.SETS["test2"]["required"]
    assert req["AI_ESCALATION"] == "off" and req["TYRE_REVIEW"] == "on"
    assert final_test.SETS["test"]["result"] != final_test.SETS["test2"]["result"]  # first result never overwritten
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    final_test.use_set("test2")
    try:
        for k, v in req.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("TYRE_REVIEW", "off")
        assert any("TYRE_REVIEW" in p for p in final_test.preflight())
    finally:
        final_test.use_set("test")


def test_test2_plan_is_fixed_and_uses_only_the_new_photos():
    import csv
    from collections import Counter
    import generate_test2
    with open(ROOT / "data/labels_test2.csv", newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    plan1, spare1 = generate_test2.plan(labels)
    plan2, _ = generate_test2.plan(labels)
    assert [(s, r["photo"]) for s, r in plan1] == [(s, r["photo"]) for s, r in plan2]  # same seed, same plan
    mix = Counter(s for s, _ in plan1)
    assert mix == {"honest": 7, "excluded_tyre": 3, "inflated_estimate": 3, "exaggerated_damage": 2,
                   "new_policy": 1, "lapsed_policy": 1, "third_party_only": 1}
    with open(ROOT / "data/labels.csv", newline="", encoding="utf-8") as f:
        old_photos = {r["photo"] for r in csv.DictReader(f)}
    assert not {r["photo"] for _, r in plan1} & old_photos and spare1
    assert {"car_080.jpg", "car_088.jpg"} <= {r["photo"] for s, r in plan1 if s == "honest"}


def test_non_dev_claims_are_refused_by_single_claim_tools():
    import run_graph
    src = (ROOT / "run_graph.py").read_text(encoding="utf-8") + (ROOT / "run_intake.py").read_text(encoding="utf-8")
    assert src.count('.get("split", "dev") != "dev"') == 2
    assert run_graph  # imported fine


def test_decisions_are_recorded_before_test2():
    lines = (ROOT / "outputs/policy_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    text = " ".join(lines)
    assert "AI_ESCALATION=off" in text and "TYRE_REVIEW=on" in text and "test2" in text


# ---------------------------------------------------------------------------
# third held-out set
# ---------------------------------------------------------------------------
def test_test3_plan_uses_only_photos_test2_did_not_use():
    import csv
    from collections import Counter
    import generate_test2
    with open(ROOT / "data/labels_test2.csv", newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    generate_test2.use_set("test2")
    p2, spare2 = generate_test2.plan(labels)
    used2 = {r["photo"] for _, r in p2} | {spare2[0]["photo"]}
    generate_test2.use_set("test3")
    try:
        rest = [r for r in labels if r["photo"] not in used2]
        p3, spare3 = generate_test2.plan(rest)
        assert not {r["photo"] for _, r in p3} & used2
        assert Counter(s for s, _ in p3) == {"honest": 9, "inflated_estimate": 4, "exaggerated_damage": 2,
                                             "new_policy": 1, "lapsed_policy": 1, "third_party_only": 1}
        assert spare3  # one photo left for the duplicate-invoice claim
    finally:
        generate_test2.use_set("test2")


def test_test3_must_test_the_same_system_as_test2():
    import final_test
    assert final_test.SETS["test3"]["same_system_as"] == "test2"
    assert final_test.SETS["test3"]["required"] == final_test.SETS["test2"]["required"]


def test_combined_refuses_different_systems(tmp_path, monkeypatch):
    import final_test
    base = {"complete": True, "rows": [{"scenario": "honest", "expected": "approve", "got": "approve",
                                        "photo_type_ok": True, "photo_within_one": True}]}
    for name, fp in (("a", "111"), ("b", "222")):
        (tmp_path / f"{name}.json").write_text(json.dumps({**base, "fingerprint": fp}), encoding="utf-8")
    monkeypatch.setattr(final_test, "SETS", {n: {"result": tmp_path / f"{n}.json"} for n in ("a", "b")})
    with pytest.raises(SystemExit):
        final_test.combined(["a", "b"])
