"""Unit tests for the intake validation rules. Run with:  pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake import validate  # noqa: E402
from schemas import ClaimForm, EstimateLine, PhotoAssessment, RepairEstimate  # noqa: E402


def make(form_over=None, est_over=None, photo_over=None):
    form = dict(claim_number="C999", date_submitted="2026-06-10", claimant_name="Ali Khan",
                cnic="12345-1234567-1", phone="0300-1234567", policy_number="PMC-2026-10001",
                vehicle="Honda City", registration_number="LE-12-345", incident_date="2026-06-08",
                location="Gulberg, Lahore", description="Rear bumper dent.", amount_claimed=50000.0)
    est = dict(workshop="City Car Care", invoice_number="INV-111111", estimate_date="2026-06-09",
               customer_name="Ali Khan", vehicle="Honda City", registration_number="LE-12-345",
               line_items=[EstimateLine(item="Parts", amount=30000),
                           EstimateLine(item="Labour", amount=20000)], total=50000.0)
    photo = dict(is_vehicle_photo=True, damaged_part="rear bumper", damage_type="dent",
                 severity="moderate", all_damages=[], description="Dent.",
                 image_quality_ok=True)
    form.update(form_over or {}); est.update(est_over or {}); photo.update(photo_over or {})
    return validate(ClaimForm(**form), RepairEstimate(**est), PhotoAssessment(**photo))


def codes(result):
    issues, _ = result
    return {i.code for i in issues}


def test_clean_claim_has_no_issues():
    issues, missing = make()
    assert issues == [] and missing == []


def test_missing_policy_number():
    _, missing = make(form_over={"policy_number": None})
    assert "claim_form.policy_number" in missing


def test_future_incident():
    assert "incident_in_future" in codes(make(form_over={"incident_date": "2027-01-01",
                                                          "date_submitted": "2027-01-02"}))


def test_submitted_before_incident():
    assert "submitted_before_incident" in codes(make(form_over={"date_submitted": "2026-06-01"}))


def test_amount_mismatch():
    assert "amount_mismatch" in codes(make(form_over={"amount_claimed": 90000.0}))


def test_lines_dont_add_up():
    assert "estimate_lines_dont_add_up" in codes(make(est_over={"total": 70000.0},
                                                     form_over={"amount_claimed": 70000.0}))


def test_name_mismatch_ignores_case_and_spaces():
    assert "name_mismatch" not in codes(make(est_over={"customer_name": "ALI  KHAN"}))
    assert "name_mismatch" in codes(make(est_over={"customer_name": "Usman Raza"}))


def test_photo_not_a_vehicle():
    assert "photo_not_vehicle" in codes(make(photo_over={"is_vehicle_photo": False}))
