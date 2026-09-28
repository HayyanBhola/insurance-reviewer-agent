"""
Fraud agent: warning signs that a human investigator should look at.

All checks are plain Python rules (no AI): they are free, instant, explainable,
and exactly the kind of thing a rule is better at than an LLM.

Signals
-------
photo_reused         same photo (perceptual fingerprint) in an EARLIER claim by someone else  -> high
invoice_reused       same invoice number in an EARLIER claim                                   -> high
new_policy           incident within 7 days of the policy start date                           -> medium
late_report          claim submitted more than 30 days after the incident                      -> low
repeat_claimant      same claimant has other claims in the last 12 months                      -> low
"""

from datetime import timedelta

import imagehash

from data_access import get_policy, parse_date
from findings import Flag, FraudFinding

PHOTO_HASH_MAX_DISTANCE = 6   # 0 = identical; small numbers = same photo resized/re-saved
NEW_POLICY_DAYS = 7
LATE_REPORT_DAYS = 30


def _earlier(other, me):
    """True if claim `other` was submitted before claim `me` (ties broken by claim id)."""
    a, b = parse_date(other["submitted"]), parse_date(me["submitted"])
    if a and b and a != b:
        return a < b
    return other["_id"] < me["_id"]


def check_fraud(claim_id, intake, history) -> FraudFinding:
    signals = []
    me = {**history.get(claim_id, {}), "_id": claim_id}
    form = intake.claim_form

    # --- compare with every EARLIER claim in the insurer's history -------------
    my_hash = imagehash.hex_to_hash(me["photo_hash"]) if me.get("photo_hash") else None
    for other_id, other in history.items():
        if other_id == claim_id:
            continue
        other = {**other, "_id": other_id}
        if not _earlier(other, me):
            continue
        same_person = (other.get("claimant") or "").lower() == (form.claimant_name or "").lower()

        if my_hash is not None and other.get("photo_hash"):
            dist = my_hash - imagehash.hex_to_hash(other["photo_hash"])
            if dist <= PHOTO_HASH_MAX_DISTANCE and not same_person:
                signals.append(Flag(code="photo_reused", severity="high",
                                    message=f"Photo matches the photo in earlier claim {other_id} "
                                            f"by {other.get('claimant')} (fingerprint distance {dist})."))

        inv = (intake.estimate.invoice_number or "").strip().upper()
        if inv and inv == (other.get("invoice_number") or "").strip().upper():
            signals.append(Flag(code="invoice_reused", severity="high",
                                message=f"Invoice {inv} was already used in earlier claim {other_id} "
                                        f"({other.get('claimant')})."))

        if same_person:
            d_other, d_me = parse_date(other.get("incident")), parse_date(form.incident_date)
            if d_other and d_me and abs((d_me - d_other).days) <= 365:
                signals.append(Flag(code="repeat_claimant", severity="low",
                                    message=f"Same claimant filed claim {other_id} within 12 months."))

    # --- policy timing ------------------------------------------------------------
    policy = get_policy(form.policy_number)
    incident, submitted = parse_date(form.incident_date), parse_date(form.date_submitted)
    if policy and incident:
        start = parse_date(policy["start_date"])
        days = (incident - start).days
        if 0 <= days <= NEW_POLICY_DAYS:
            signals.append(Flag(code="new_policy", severity="medium",
                                message=f"Incident happened only {days} day(s) after the policy started "
                                        f"({start})."))
    if incident and submitted and submitted - incident > timedelta(days=LATE_REPORT_DAYS):
        signals.append(Flag(code="late_report", severity="low",
                            message=f"Claim reported {(submitted - incident).days} days after the incident."))

    sev = {s.severity for s in signals}
    risk = "high" if "high" in sev else ("medium" if "medium" in sev else "low")
    return FraudFinding(risk=risk, signals=signals)
