"""Photo benchmark + grounded (v2) photo schema. No API calls. Run: pytest -q"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_splits_are_fixed_disjoint_and_exclude_test_photos():
    import eval_photos
    s = eval_photos.photo_splits()
    assert s == eval_photos.photo_splits()                      # deterministic
    assert not set(s["tune"]) & set(s["check"])                 # disjoint
    test_photos = {t["photo_source"] for t in eval_photos.TRUTH.values() if t["split"] == "test"}
    assert not (set(s["tune"]) | set(s["check"])) & test_photos  # no test leakage


def test_low_confidence_damage_is_dropped():
    from schemas import GroundedPhotoAssessment
    g = GroundedPhotoAssessment(
        is_vehicle_photo=True, damaged_part="headlight", damage_type="lamp broken", severity="moderate",
        description="Broken headlight.", image_quality_ok=True, all_damages=[
            {"part": "headlight", "damage_type": "lamp broken", "severity": "moderate",
             "evidence": "lens cracked", "confidence": "high"},
            {"part": "fender", "damage_type": "dent", "severity": "severe",
             "evidence": "maybe a shadow", "confidence": "low"}])
    a = g.to_assessment()
    assert [d.part for d in a.all_damages] == ["headlight"]


def test_phantom_damage_cannot_raise_the_price_ceiling():
    """The C012 pattern: a low-confidence severe dent must not make an inflated bill look normal.
    Uses the TRUE damage label (not a saved AI reading), so it tests only the mechanism."""
    from data_access import load_intake
    from evidence_agent import cost_check
    from schemas import GroundedPhotoAssessment
    truth = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
    cid, t = next((c, t) for c, t in truth.items()   # dev claims only: test claims are off limits
                  if t["split"] == "dev" and t["scenario"] == "inflated_estimate"
                  and (ROOT / f"outputs/intake/{c}.json").exists())
    real = t["true_damage"]
    r = load_intake(cid)

    def photo(extra_confidence):
        return GroundedPhotoAssessment(
            is_vehicle_photo=True, damaged_part=real["part"], damage_type=real["damage_type"],
            severity=real["severity"], description="", image_quality_ok=True, all_damages=[
                {"part": real["part"], "damage_type": real["damage_type"], "severity": real["severity"],
                 "evidence": "seen", "confidence": "high"},
                {"part": "fender", "damage_type": "dent", "severity": "severe",
                 "evidence": "unclear", "confidence": extra_confidence}]).to_assessment()

    # a CONFIDENT severe dent would hide the inflated bill (that's what happened in v1) ...
    assert cost_check(r.model_copy(update={"photo": photo("high")}))[2] == "within"
    # ... but an UNSURE one is dropped, so the inflated bill is caught
    assert cost_check(r.model_copy(update={"photo": photo("low")}))[2] in ("above", "far_above")


def test_score_counts_over_and_under_reads():
    import eval_photos
    photo = eval_photos.photo_splits()["tune"][0]
    lab = eval_photos.LABELS[photo]
    pred = {"severity": "severe" if lab["severity"] != "severe" else "minor", "damage_type": lab["damage_type"],
            "damaged_part": lab["part"], "all_damages": []}
    r = eval_photos.score(photo, pred)
    assert r["type_ok"] and not r["sev_ok"] and (r["over"] or r["under"])


# ---------------------------------------------------------------------------
# v3: wear-and-tear rule
# ---------------------------------------------------------------------------
def test_v3_is_exactly_v1_plus_the_wear_rule():
    import intake
    from schemas import PhotoAssessment
    assert intake.PHOTO_SYSTEM_V3 == intake.PHOTO_SYSTEM + intake.WEAR_RULE  # one change only
    schema, msgs = intake.photo_request(ROOT / "data/photos/car_010.jpg", "v3")
    assert schema is PhotoAssessment and "PRE-EXISTING WEAR" in msgs[0].content
    _, v1 = intake.photo_request(ROOT / "data/photos/car_010.jpg", "v1")
    assert "PRE-EXISTING WEAR" not in v1[0].content
    with pytest.raises(ValueError):
        intake.photo_request(ROOT / "data/photos/car_010.jpg", "v9")


def test_extra_photos_are_unused_tyre_photos_and_final_is_fresh():
    import eval_photos
    s = eval_photos.photo_splits()
    used = {t["photo_source"] for t in eval_photos.TRUTH.values()}
    assert s["extra"] and not set(s["extra"]) & used
    assert all(eval_photos.is_tyre_only(eval_photos.LABELS[p]) for p in s["extra"])
    assert s["final"] == s["check"] + s["extra"] and not set(s["final"]) & set(s["tune"])


def _pred(types, main="tire flat"):
    return {"severity": "moderate", "damage_type": main, "damaged_part": "front tire",
            "all_damages": [{"part": "p", "damage_type": t, "severity": "moderate"} for t in types]}


def test_tyre_only_and_damage_found_scoring():
    import eval_photos
    tyre_photo = eval_photos.photo_splits()["extra"][0]
    assert eval_photos.score(tyre_photo, _pred(["tire flat"]))["tyre_only_ok"]
    assert not eval_photos.score(tyre_photo, _pred(["tire flat", "dent"]))["tyre_only_ok"]
    assert eval_photos.score(tyre_photo, _pred(["dent", "tire flat"], main="dent"))["found"]
    assert not eval_photos.score(tyre_photo, _pred(["dent"], main="dent"))["found"]


def test_body_damage_guard():
    import eval_photos
    mixed = next(p for p, lab in eval_photos.LABELS.items()
                 if lab["damage_type"] == "tire flat" and eval_photos.has_body_damage(lab))
    assert eval_photos.score(mixed, _pred(["tire flat", "scratch"]))["body_kept"]
    assert not eval_photos.score(mixed, _pred(["tire flat"]))["body_kept"]      # body damage hidden
    tyre = eval_photos.photo_splits()["extra"][0]
    assert not eval_photos.score(tyre, _pred(["tire flat"]))["body_label"]      # not counted there


def test_keep_rule_needs_every_condition():
    import eval_photos
    old = {"tyre_only": 0.25, "damage_found": 0.90, "type_accuracy": 0.70, "within_one": 0.95,
           "body_kept": 0.95}
    assert eval_photos.decide(old, {**old, "tyre_only": 0.75})[0]                         # better tyres
    assert not eval_photos.decide(old, {**old})[0]                                        # no gain
    assert not eval_photos.decide(old, {**old, "tyre_only": 0.75, "damage_found": 0.80})[0]  # hides damage
    assert not eval_photos.decide(old, {**old, "tyre_only": 0.75, "type_accuracy": 0.60})[0]
    assert eval_photos.decide(old, {**old, "tyre_only": 0.75, "within_one": 0.92})[0]     # within noise
    assert not eval_photos.decide(old, {**old, "tyre_only": 0.75, "body_kept": 0.85})[0]  # hides body damage


def test_why_tyre_only_reading_matters_for_cover():
    """A flat tyre alone -> exclusion may apply; tyre + body damage -> 'vehicle damaged at the
    same time', so the tyre is covered. That is why wear read as a dent broke the tyre claims."""
    from coverage_agent import check_coverage
    from data_access import load_intake
    from schemas import PhotoAssessment
    truth = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
    cid = next(c for c, t in truth.items() if t["scenario"] == "excluded_tyre" and t["split"] == "dev"
               and (ROOT / f"outputs/intake/{c}.json").exists())
    r = load_intake(cid)

    def with_photo(types):
        p = PhotoAssessment.model_validate({**_pred(types), "is_vehicle_photo": True,
                                            "description": "", "image_quality_ok": True})
        return r.model_copy(update={"photo": p})

    assert check_coverage(with_photo(["tire flat"]), None, use_llm=False).status == "needs_review"
    assert check_coverage(with_photo(["tire flat", "dent"]), None, use_llm=False).status == "covered"


def _add_non_ascii(folder):
    """Real model output contains characters like '—' and 'é'; Windows' default encoding
    (cp1252) can't read them. Put some in, so encoding bugs show up on every machine."""
    for f in folder.glob("C*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["photo"]["description"] += " — pre-existing wear (rusté)"
        f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# refresh_photos.py: only photos change, with a backup, on a temporary copy
# ---------------------------------------------------------------------------
def test_refresh_changes_only_photos_backs_up_and_restores(monkeypatch, tmp_path):
    import shutil
    import refresh_photos
    from schemas import IntakeResult, PhotoAssessment

    intake_dir = tmp_path / "intake"
    shutil.copytree(ROOT / "outputs/intake", intake_dir)
    _add_non_ascii(intake_dir)
    monkeypatch.setattr(refresh_photos, "INTAKE", intake_dir)
    monkeypatch.setattr(refresh_photos, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(refresh_photos, "MARKER", intake_dir / "_photo_prompt.json")

    tyre_only = PhotoAssessment.model_validate({**_pred(["tire flat"]), "is_vehicle_photo": True,
                                                "description": "pre-existing wear: rust", "image_quality_ok": True})
    monkeypatch.setattr(refresh_photos, "assess_photo", lambda path, version=None: tyre_only)

    cid = sorted(f.stem for f in intake_dir.glob("C*.json")
                 if json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))[f.stem]["split"] == "dev")[0]
    before = IntakeResult.model_validate_json((intake_dir / f"{cid}.json").read_text(encoding="utf-8"))
    refresh_photos.refresh("v3", dry_run=False)
    after = IntakeResult.model_validate_json((intake_dir / f"{cid}.json").read_text(encoding="utf-8"))

    assert after.photo == tyre_only                                   # photo replaced
    assert after.claim_form == before.claim_form and after.estimate == before.estimate  # forms untouched
    assert json.loads((intake_dir / "_photo_prompt.json").read_text(encoding="utf-8"))["photo_prompt"] == "v3"
    assert any((tmp_path / "backups").iterdir())                      # backup made first

    refresh_photos.restore()
    restored = IntakeResult.model_validate_json((intake_dir / f"{cid}.json").read_text(encoding="utf-8"))
    assert restored.photo == before.photo                             # undo works


def test_refresh_dry_run_writes_nothing(monkeypatch, tmp_path):
    import shutil
    import refresh_photos
    from schemas import PhotoAssessment
    intake_dir = tmp_path / "intake"
    shutil.copytree(ROOT / "outputs/intake", intake_dir)
    _add_non_ascii(intake_dir)
    monkeypatch.setattr(refresh_photos, "INTAKE", intake_dir)
    monkeypatch.setattr(refresh_photos, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(refresh_photos, "MARKER", intake_dir / "_photo_prompt.json")
    p = PhotoAssessment.model_validate({**_pred(["tire flat"]), "is_vehicle_photo": True,
                                        "description": "", "image_quality_ok": True})
    monkeypatch.setattr(refresh_photos, "assess_photo", lambda path, version=None: p)
    snapshot = {f.name: f.read_bytes() for f in intake_dir.iterdir()}
    refresh_photos.refresh("v3", dry_run=True)
    assert {f.name: f.read_bytes() for f in intake_dir.iterdir()} == snapshot
    assert not (tmp_path / "backups").exists()


def test_decision_change_labels():
    from refresh_photos import _judge
    assert _judge("approve", "investigate", "deny") == "safer"      # tyre claim, rules-only
    assert _judge("investigate", "deny", "deny") == "better"
    assert _judge("approve", "investigate", "approve") == "WORSE"   # honest claim now held up
    assert _judge("investigate", "approve", "deny") == "WORSE"


def test_out_of_credit_stops_instead_of_waiting(monkeypatch):
    import eval_photos
    import intake
    from llm import out_of_credit
    assert out_of_credit(RuntimeError("Error 429: insufficient_quota - You exceeded your current quota"))
    assert not out_of_credit(RuntimeError("429 RESOURCE_EXHAUSTED rate limit"))
    monkeypatch.setattr(intake, "assess_photo", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("429 insufficient_quota")))
    monkeypatch.setattr(eval_photos.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("waited")))
    with pytest.raises(SystemExit):
        eval_photos.collect("v3", eval_photos.photo_splits()["tune"][:2])
