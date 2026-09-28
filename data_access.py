"""
Read-only access to the insurer's "databases":
  - policy records   (data/policy_records.csv)
  - repair costs     (data/repair_costs.csv)
  - intake results   (outputs/intake/<claim>.json, produced in Phase 2)
  - claims history   (every claim ever submitted: invoice numbers, dates, photo fingerprints)

No AI here, so everything in this file is free and instant.
"""

import csv
import json
import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from schemas import IntakeResult

DATA = Path("data")
CLAIMS_DIR = DATA / "claims"
INTAKE_DIR = Path("outputs/intake")
HISTORY_FILE = Path("outputs/claims_history.json")
TODAY = date.fromisoformat(os.getenv("SIMULATED_TODAY", "2026-09-28"))


def parse_date(value):
    try:
        return date.fromisoformat(value) if value else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Policies and repair costs
# ---------------------------------------------------------------------------
@lru_cache
def _policies():
    with open(DATA / "policy_records.csv", newline="", encoding="utf-8") as f:
        return {row["policy_number"].strip().upper(): row for row in csv.DictReader(f)}


def get_policy(policy_number):
    """Look up a policy by number. Returns a dict or None."""
    if not policy_number:
        return None
    return _policies().get(policy_number.strip().upper())


@lru_cache
def repair_costs():
    """{(damage_type, severity): (min_pkr, max_pkr)}"""
    with open(DATA / "repair_costs.csv", newline="", encoding="utf-8") as f:
        return {(r["damage_type"], r["severity"]): (float(r["min_pkr"]), float(r["max_pkr"]))
                for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Intake results from Phase 2 (so we never pay for intake twice)
# ---------------------------------------------------------------------------
def load_intake(claim_id) -> IntakeResult:
    path = INTAKE_DIR / f"{claim_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No intake result for {claim_id}. Run: python run_intake.py {claim_id}")
    return IntakeResult.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Claims history: the insurer's record of ALL claims ever submitted.
# Built once from the claim documents with simple text parsing (no AI) and cached.
# ---------------------------------------------------------------------------
def _field(text, label):
    m = re.search(re.escape(label) + r"\s*\n\s*(.+)", text)
    return m.group(1).strip() if m else None


def _pdf_text(path):
    return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)


def build_claims_history(force=False):
    """One record per claim folder: claimant, dates, invoice number, photo fingerprint."""
    import imagehash
    from PIL import Image

    if HISTORY_FILE.exists() and not force:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

    history = {}
    for folder in sorted(p for p in CLAIMS_DIR.iterdir() if p.is_dir()):
        form = _pdf_text(folder / "claim_form.pdf")
        est = _pdf_text(folder / "repair_estimate.pdf")
        history[folder.name] = {
            "claimant": _field(form, "Claimant name"),
            "submitted": _field(form, "Date submitted"),
            "incident": _field(form, "Date of incident"),
            "invoice_number": _field(est, "Invoice / estimate no."),
            "workshop": _field(est, "Workshop"),
            "photo_hash": str(imagehash.phash(Image.open(folder / "photo_1.jpg"))),
        }
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
