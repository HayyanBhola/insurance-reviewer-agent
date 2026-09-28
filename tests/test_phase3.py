"""Unit tests for Phase 3 (no AI calls, free). Run with:  pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fraud_agent  # noqa: E402
from coverage_agent import quote_in_text, verify_citations  # noqa: E402
from evidence_agent import cost_check  # noqa: E402
from findings import (ClauseCitation, CoverageFinding, CoverageJudgment,  # noqa: E402
                      EvidenceFinding, Flag, FraudFinding)
from policy_index import _is_heading, rrf, tokenize  # noqa: E402
from run_specialists import preliminary_decision  # noqa: E402
from schemas import (ClaimForm, DamageItem, IntakeResult, PhotoAssessment,  # noqa: E402
                     RepairEstimate)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def intake(amount=40000.0, dtype="dent", sev="moderate", invoice="INV-1", name="Ali Khan",
           incident="2026-06-10", submitted="2026-06-12", policy="PMC-1"):
    return IntakeResult(
        claim_id="C999",
        claim_form=ClaimForm(claim_number="C999", date_submitted=submitted, claimant_name=name,
                             cnic=None, phone=None, policy_number=policy, vehicle=None,
                             registration_number=None, incident_date=incident, location=None,
                             description="x", amount_claimed=amount),
        estimate=RepairEstimate(workshop=None, invoice_number=invoice, estimate_date=None,
                                customer_name=name, vehicle=None, registration_number=None,
                                line_items=[], total=amount),
        photo=PhotoAssessment(is_vehicle_photo=True,
                              all_damages=[DamageItem(part="door", damage_type=dtype, severity=sev)],
                              damaged_part="door", damage_type=dtype, severity=sev,
                              description="x", image_quality_ok=True),
        issues=[], missing_fields=[], ready_for_review=True)


def cov(status="covered", method="rules+rag"):
    return CoverageFinding(status=status, policy_found=True, policy_active_on_incident=True,
                           coverage_type="comprehensive", deductible=5000, sum_insured=1e6,
                           policy_wording="x.pdf", wording_searchable=True,
                           reasons=["Policy active.", "Reason."], citations=[], method=method)


def ev(flags=()):
    return EvidenceFinding(photo_summary="x", claimed_amount=1, typical_range_pkr=(1, 2),
                           cost_ratio=0.5, cost_status="within", description_consistent=True,
                           description_note="", flags=list(flags))


# ---------------------------------------------------------------------------
# policy search
# ---------------------------------------------------------------------------
def test_tokenize_maps_spelling_variants():
    assert "tyre" in tokenize("Flat tire")
    assert "tyre" in tokenize("damage to tyres")
    assert "windscreen" in tokenize("cracked windshield")


def test_heading_detection():
    assert _is_heading("SECTION A: LOSS OR DAMAGE TO YOUR OWN CAR")
    assert _is_heading("Section II - Liability")
    assert not _is_heading("We will pay for loss or damage to your car.")


def test_rrf_rewards_items_ranked_well_in_both_lists():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


# ---------------------------------------------------------------------------
# citation verification (anti-hallucination)
# ---------------------------------------------------------------------------
CHUNK = {"id": "p:p1:c0", "policy_file": "p.pdf", "page": 1, "section": "S",
         "text": "Events We Do Not Cover: Any damage to the tyre(s) of Your Car unless other\n"
                 "parts of Your Car are also damaged at the same time."}


def test_real_quote_is_verified_even_across_line_breaks():
    assert quote_in_text("Any damage to the tyre(s) of Your Car unless other parts of Your Car "
                         "are also damaged at the same time.", CHUNK["text"])


def test_invented_quote_is_rejected():
    assert not quote_in_text("Tyres are never covered under this policy in any situation.",
                             CHUNK["text"])


def test_invented_clause_id_is_rejected():
    j = CoverageJudgment(decision="not_covered", reasoning="r",
                         citations=[ClauseCitation(chunk_id="fake:id", quote="Any damage to the tyre(s)")])
    assert verify_citations(j, [CHUNK])[0].verified is False


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def test_cost_within_and_far_above():
    # moderate dent typical max is 80,000 PKR in data/repair_costs.csv
    assert cost_check(intake(amount=60000))[2] == "within"
    assert cost_check(intake(amount=300000))[2] == "far_above"


# ---------------------------------------------------------------------------
# fraud
# ---------------------------------------------------------------------------
def _history(**overrides):
    base = {"claimant": "Someone Else", "submitted": "2026-05-01", "incident": "2026-04-28",
            "invoice_number": "INV-OLD", "photo_hash": "ffffffffffffffff"}
    me = {"claimant": "Ali Khan", "submitted": "2026-06-12", "incident": "2026-06-10",
          "invoice_number": "INV-1", "photo_hash": "0000000000000000"}
    base.update(overrides)
    return {"C001": base, "C999": me}


def _no_policy(monkeypatch):
    monkeypatch.setattr(fraud_agent, "get_policy", lambda n: None)


def test_reused_invoice_in_earlier_claim(monkeypatch):
    _no_policy(monkeypatch)
    f = fraud_agent.check_fraud("C999", intake(invoice="INV-OLD"), _history())
    assert any(s.code == "invoice_reused" for s in f.signals) and f.risk == "high"


def test_reused_photo_only_flags_the_later_claim(monkeypatch):
    _no_policy(monkeypatch)
    later = fraud_agent.check_fraud("C999", intake(), _history(photo_hash="0000000000000000"))
    assert any(s.code == "photo_reused" for s in later.signals)
    # same photos, but now the other claim is LATER than ours -> we are the original
    earlier = fraud_agent.check_fraud("C999", intake(), _history(photo_hash="0000000000000000",
                                                                 submitted="2026-07-01"))
    assert not any(s.code == "photo_reused" for s in earlier.signals)


def test_new_policy_signal(monkeypatch):
    monkeypatch.setattr(fraud_agent, "get_policy", lambda n: {"start_date": "2026-06-07"})
    f = fraud_agent.check_fraud("C999", intake(), _history())
    assert any(s.code == "new_policy" for s in f.signals) and f.risk == "medium"


def test_clean_claim_has_low_risk(monkeypatch):
    _no_policy(monkeypatch)
    assert fraud_agent.check_fraud("C999", intake(), _history()).risk == "low"


# ---------------------------------------------------------------------------
# preliminary decision
# ---------------------------------------------------------------------------
def test_decisions():
    low = FraudFinding(risk="low", signals=[])
    high = FraudFinding(risk="high", signals=[Flag(code="x", severity="high", message="m")])
    assert preliminary_decision(cov("covered"), ev(), low)[0] == "approve"
    assert preliminary_decision(cov("not_covered"), ev(), low)[0] == "deny"
    assert preliminary_decision(cov("needs_review"), ev(), low)[0] == "investigate"
    assert preliminary_decision(cov("covered"), ev(), high)[0] == "investigate"
    low_flag = [Flag(code="y", severity="low", message="m")]
    assert preliminary_decision(cov("covered"), ev(low_flag), low)[0] == "approve"
