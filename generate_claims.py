"""
Synthetic claim generator for the Insurance Claims Reviewer project.

Reads:   data/labels.csv, data/photos/
Writes:  data/claims/C001 ... C0NN/   (claim_form.pdf, repair_estimate.pdf, photo_1.jpg)
         data/policy_records.csv     (the insurer's policy database)
         data/repair_costs.csv       (typical repair cost ranges, PKR)
         data/ground_truth.json      (correct answer + planted issues per claim)
         data/claims_summary.csv     (one row per claim, easy to scan)

No LLM calls: everything is generated with templates and a fixed random seed,
so running it again gives exactly the same claims.
"""

import csv
import json
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEED = 42
DATA = Path("data")
PHOTOS = DATA / "photos"
CLAIMS = DATA / "claims"
TODAY = date(2026, 9, 28)  # "current date" of the simulated insurer

random.seed(SEED)

# ---------------------------------------------------------------------------
# Reference data (all fictional)
# ---------------------------------------------------------------------------
FIRST_NAMES = ["Ali", "Ahmed", "Usman", "Bilal", "Hamza", "Fatima", "Ayesha", "Zainab",
               "Hassan", "Omar", "Sana", "Hina", "Imran", "Kashif", "Mariam", "Saad",
               "Nida", "Faisal", "Rabia", "Tariq", "Sadia", "Junaid", "Amna", "Waqas"]
LAST_NAMES = ["Khan", "Ahmed", "Malik", "Hussain", "Qureshi", "Sheikh", "Butt", "Raza",
              "Siddiqui", "Chaudhry", "Iqbal", "Mirza", "Abbasi", "Javed", "Aslam"]
CITIES = {
    "Lahore": ["Gulberg", "DHA Phase 5", "Model Town", "Johar Town", "Mall Road"],
    "Karachi": ["Clifton", "Shahrah-e-Faisal", "Gulshan-e-Iqbal", "PECHS", "Korangi"],
    "Islamabad": ["F-7 Markaz", "Blue Area", "G-11", "Srinagar Highway", "I-8"],
    "Rawalpindi": ["Saddar", "Murree Road", "Bahria Town", "Chaklala"],
    "Faisalabad": ["D Ground", "Canal Road", "Jaranwala Road"],
}
CITY_CODE = {"Lahore": "LE", "Karachi": "KHI", "Islamabad": "ICT", "Rawalpindi": "RIR",
             "Faisalabad": "FDA"}
VEHICLES = ["Suzuki Alto", "Suzuki Cultus", "Suzuki Wagon R", "Toyota Corolla",
            "Toyota Yaris", "Honda City", "Honda Civic", "Kia Sportage", "Hyundai Tucson",
            "Toyota Fortuner", "Changan Alsvin", "MG HS"]
WORKSHOPS = ["Al-Madina Auto Works", "Star Motors Body Shop", "City Car Care",
             "Ittefaq Autos", "Prime Auto Garage", "Khan Denting & Painting"]
POLICY_WORDINGS = ["jubilee_private_car.pdf", "icici_private_car.pdf", "etiqa_private_car.pdf"]

# Typical repair cost ranges in PKR by (damage_type, severity). Fictional but plausible.
REPAIR_COSTS = {
    ("scratch", "minor"): (8000, 20000),
    ("scratch", "moderate"): (18000, 45000),
    ("scratch", "severe"): (40000, 90000),
    ("dent", "minor"): (12000, 30000),
    ("dent", "moderate"): (30000, 80000),
    ("dent", "severe"): (90000, 300000),
    ("crack", "minor"): (15000, 35000),
    ("crack", "moderate"): (30000, 70000),
    ("crack", "severe"): (60000, 150000),
    ("lamp broken", "minor"): (10000, 25000),
    ("lamp broken", "moderate"): (25000, 70000),
    ("lamp broken", "severe"): (60000, 180000),
    ("glass shatter", "minor"): (20000, 45000),
    ("glass shatter", "moderate"): (40000, 90000),
    ("glass shatter", "severe"): (60000, 160000),
    ("tire flat", "minor"): (3000, 8000),
    ("tire flat", "moderate"): (8000, 25000),
    ("tire flat", "severe"): (20000, 60000),
}

# How a claimant would describe each damage type
DESCRIPTIONS = {
    "scratch": "While parking, another vehicle scraped the {part}, leaving {sev} scratches on the paint.",
    "dent": "Another car hit my vehicle, causing a {sev} dent on the {part}.",
    "crack": "My car struck a raised divider, and the {part} cracked. The damage is {sev}.",
    "lamp broken": "A motorcycle collided with my car and broke the {part}. The damage is {sev}.",
    "glass shatter": "A stone hit the car and the {part} shattered. The damage is {sev}.",
    "tire flat": "The {part} burst while driving and went completely flat.",
}
SEVERITY_WORDS = {"minor": "minor", "moderate": "noticeable", "severe": "serious"}

# Exaggerated version: claims far worse damage than the photo shows
EXAGGERATIONS = [
    "A truck hit my car at high speed. The {part} is completely destroyed, both headlights are "
    "smashed, the bonnet is crushed and the radiator is leaking.",
    "My car was in a major collision. The {part} was torn off, the windshield shattered and the "
    "front suspension is damaged.",
    "The car rolled over on the motorway. The {part}, roof and both doors on that side are "
    "badly crushed and several windows are broken.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fake_person():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def fake_cnic():
    return f"{random.randint(10000, 99999)}-{random.randint(1000000, 9999999)}-{random.randint(1, 9)}"


def fake_phone():
    return f"03{random.randint(0, 4)}{random.randint(0, 9)}-{random.randint(1000000, 9999999)}"


def fake_reg(city):
    return f"{CITY_CODE[city]}-{random.randint(10, 99)}-{random.randint(100, 9999)}"


def rand_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))


def pkr(n):
    return f"PKR {n:,.0f}"


def estimate_lines(damage, part, total):
    """Split a total into parts / paint / labour lines that sum exactly to total."""
    parts_share = random.uniform(0.40, 0.60)
    paint_share = random.uniform(0.10, 0.25) if damage in ("scratch", "dent", "crack") else 0
    parts = round(total * parts_share, -2)
    paint = round(total * paint_share, -2)
    labour = total - parts - paint
    lines = [(f"{part.title()} - replacement / repair parts", parts)]
    if paint:
        lines.append((f"{part.title()} - denting & painting", paint))
    lines.append(("Labour charges", labour))
    return lines


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
STYLES = getSampleStyleSheet()


def _table(rows, col_widths):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    return t


def write_claim_form(path, c):
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    p = lambda s: Paragraph(s, STYLES["BodyText"])
    rows = [
        ["Claim number", c["claim_id"]],
        ["Date submitted", c["submitted_date"]],
        ["Claimant name", c["claimant"]],
        ["CNIC", c["cnic"]],
        ["Phone", c["phone"]],
        ["Policy number", c["policy_number"]],
        ["Vehicle", c["vehicle"]],
        ["Registration no.", c["registration"]],
        ["Date of incident", c["incident_date"]],
        ["Location", c["location"]],
        ["Description of incident", p(c["description"])],
        ["Amount claimed", pkr(c["amount_claimed"])],
        ["Attachments", "photo_1.jpg, repair_estimate.pdf"],
    ]
    story = [
        Paragraph("MOTOR INSURANCE CLAIM FORM", STYLES["Title"]),
        Paragraph("Synthetic document for a portfolio project. Not a real claim.", STYLES["Italic"]),
        Spacer(1, 12),
        _table(rows, [140, 330]),
        Spacer(1, 18),
        p("I declare that the information given above is true and complete."),
        p(f"Signature: {c['claimant']}"),
    ]
    doc.build(story)


def write_estimate(path, c):
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    header = [
        ["Workshop", c["workshop"]],
        ["Invoice / estimate no.", c["invoice_number"]],
        ["Date", c["estimate_date"]],
        ["Customer", c["claimant"]],
        ["Vehicle", f"{c['vehicle']} ({c['registration']})"],
    ]
    items = [["Item", "Amount"]] + [[d, pkr(a)] for d, a in c["estimate_lines"]]
    items.append(["TOTAL", pkr(c["amount_claimed"])])
    it = Table(items, colWidths=[330, 140])
    it.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story = [
        Paragraph("REPAIR ESTIMATE", STYLES["Title"]),
        Paragraph("Synthetic document for a portfolio project. Not a real invoice.", STYLES["Italic"]),
        Spacer(1, 12),
        _table(header, [140, 330]),
        Spacer(1, 12),
        it,
    ]
    doc.build(story)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def load_labels():
    with open(DATA / "labels.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (PHOTOS / r["photo"]).exists()]
    if not rows:
        raise SystemExit("No labeled photos found. Check data/labels.csv and data/photos/.")
    return rows


def make_policy(policy_no, holder, city, kind="comprehensive", start=None):
    start = start or rand_date(date(2025, 10, 1), date(2026, 5, 1))
    return {
        "policy_number": policy_no,
        "holder": holder,
        "city": city,
        "coverage": kind,  # comprehensive | third_party_only
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=364)).isoformat(),
        "sum_insured": random.choice([1_500_000, 2_500_000, 3_500_000, 5_000_000, 8_000_000]),
        "deductible": random.choice([5000, 10000, 15000]),
        "policy_wording": random.choice(POLICY_WORDINGS),
    }


def plan_scenarios(labels):
    """Decide which photo gets which scenario. Returns list of (scenario, label_row)."""
    tyre_only = [r for r in labels if r["damage_type"] == "tire flat" and not r["other_damage"]]
    others = [r for r in labels if r not in tyre_only]
    random.shuffle(tyre_only)
    random.shuffle(others)

    # Tyre-only damage is excluded, so these photos are used only for "deny" claims.
    # We use 6 of them; the rest are left out to keep the decisions balanced.
    plan = [("excluded_tyre", r) for r in tyre_only[:6]]
    pool = list(others)

    quotas = [("lapsed_policy", 3), ("third_party_only", 3), ("new_policy", 3),
              ("exaggerated_damage", 5), ("inflated_estimate", 5)]
    for scenario, n in quotas:
        # exaggeration only makes sense when the real damage is small
        if scenario == "exaggerated_damage":
            picks = [r for r in pool if r["severity"] in ("minor", "moderate")
                     and r["damage_type"] != "tire flat"][:n]
        else:
            picks = pool[:n]
        for r in picks:
            pool.remove(r)
            plan.append((scenario, r))

    plan += [("honest", r) for r in pool]
    return plan


def build_claim(i, scenario, lab, policy_db, after=None):
    """after: a date; if given, the incident happens after it (used for copycat fraud)."""
    city = random.choice(list(CITIES))
    claimant = fake_person()
    policy_no = f"PMC-{2026}-{10000 + i}"
    issues, decision, reasons = [], "approve", []

    # --- policy -------------------------------------------------------------
    if scenario == "third_party_only":
        policy = make_policy(policy_no, claimant, city, kind="third_party_only")
    elif scenario == "lapsed_policy":
        # policy must have already ended before today, so start it a year+ ago
        policy = make_policy(policy_no, claimant, city,
                             start=rand_date(date(2024, 9, 1), date(2025, 8, 1)))
    elif after is not None:
        policy = make_policy(policy_no, claimant, city, start=after - timedelta(days=90))
    else:
        policy = make_policy(policy_no, claimant, city)
    policy_db.append(policy)
    p_start = date.fromisoformat(policy["start_date"])
    p_end = date.fromisoformat(policy["end_date"])

    # --- incident date ------------------------------------------------------
    if scenario == "lapsed_policy":
        incident = p_end + timedelta(days=random.randint(5, 40))
        issues.append("policy_expired_before_incident")
        decision, reasons = "deny", [f"Incident {incident} is after policy end date {p_end}."]
    elif scenario == "new_policy":
        incident = p_start + timedelta(days=random.randint(1, 6))
        issues.append("incident_within_7_days_of_policy_start")
        decision = "investigate"
        reasons = [f"Incident only {(incident - p_start).days} days after policy start."]
    elif after is not None:
        incident = after + timedelta(days=random.randint(7, 30))
    else:
        incident = rand_date(p_start + timedelta(days=30), min(p_end, TODAY) - timedelta(days=5))

    submitted = incident + timedelta(days=random.randint(1, 5))

    # --- description & amount -------------------------------------------------
    dmg, sev, part = lab["damage_type"], lab["severity"], lab["part"]
    lo, hi = REPAIR_COSTS[(dmg, sev)]
    amount = round(random.uniform(lo, hi), -2)
    description = DESCRIPTIONS[dmg].format(part=part, sev=SEVERITY_WORDS[sev])

    if scenario == "exaggerated_damage":
        description = random.choice(EXAGGERATIONS).format(part=part)
        amount = round(random.uniform(250_000, 600_000), -2)
        issues.append("description_exaggerates_photo_damage")
        decision = "investigate"
        reasons = [f"Photo shows {sev} {dmg} on {part}; claim describes a major collision."]
    elif scenario == "inflated_estimate":
        amount = round(hi * random.uniform(2.5, 4.0), -2)
        issues.append("estimate_far_above_typical_cost")
        decision = "investigate"
        reasons = [f"Estimate {pkr(amount)} vs typical {pkr(lo)}-{pkr(hi)} for {sev} {dmg}."]
    elif scenario == "excluded_tyre":
        issues.append("tyre_damage_only_excluded")
        decision = "deny"
        reasons = ["Damage is to the tyre only, with no other vehicle damage. "
                   "Standard private car wordings exclude tyre damage unless the vehicle "
                   "is damaged at the same time (verify the exact clause in the policy PDF)."]
    elif scenario == "third_party_only":
        issues.append("own_damage_not_covered_third_party_policy")
        decision = "deny"
        reasons = ["Policy is third-party only; own-vehicle damage is not covered."]

    if decision == "approve":
        reasons = [f"Photo matches description ({sev} {dmg} on {part}); policy active; "
                   f"amount within typical range."]

    claim_id = f"C{i:03d}"
    return {
        "claim_id": claim_id,
        "submitted_date": submitted.isoformat(),
        "claimant": claimant,
        "cnic": fake_cnic(),
        "phone": fake_phone(),
        "policy_number": policy_no,
        "vehicle": random.choice(VEHICLES),
        "registration": fake_reg(city),
        "incident_date": incident.isoformat(),
        "location": f"{random.choice(CITIES[city])}, {city}",
        "description": description,
        "amount_claimed": amount,
        "workshop": random.choice(WORKSHOPS),
        "invoice_number": f"INV-{random.randint(100000, 999999)}",
        "estimate_date": (incident + timedelta(days=random.randint(0, 3))).isoformat(),
        "estimate_lines": estimate_lines(dmg, part, amount),
        "photo_source": lab["photo"],
        "true_damage": {"part": part, "damage_type": dmg, "severity": sev,
                        "other_damage": lab["other_damage"]},
        "scenario": scenario,
        "expected_decision": decision,
        "planted_issues": issues,
        "reasons": reasons,
    }


def add_copycat_claims(claims, policy_db, start_index):
    """Reused-photo and duplicate-invoice fraud: copy something from an honest claim."""
    # only copy from honest claims submitted well before "today", so the copycat
    # claim can come later without landing in the future
    cutoff = (TODAY - timedelta(days=40)).isoformat()
    honest = [c for c in claims if c["scenario"] == "honest" and c["submitted_date"] <= cutoff]
    random.shuffle(honest)
    extra, i = [], start_index

    for original in honest[:3]:  # same photo, different claimant
        lab = {"photo": original["photo_source"], **original["true_damage"]}
        after = date.fromisoformat(original["submitted_date"])
        c = build_claim(i, "honest", lab, policy_db, after=after)
        c.update(scenario="reused_photo", expected_decision="investigate",
                 planted_issues=["photo_reused_from_" + original["claim_id"]],
                 reasons=[f"Photo is identical to the one in claim {original['claim_id']} "
                          f"from a different claimant."])
        extra.append(c)
        i += 1

    photo_donors = random.sample(honest[6:], 3)  # a different donor for each claim
    for original, other in zip(honest[3:6], photo_donors):  # same invoice, different claim
        lab = {"photo": other["photo_source"], **other["true_damage"]}
        after = max(date.fromisoformat(original["submitted_date"]),
                    date.fromisoformat(other["submitted_date"]))
        c = build_claim(i, "honest", lab, policy_db, after=after)
        c.update(scenario="duplicate_invoice", expected_decision="investigate",
                 invoice_number=original["invoice_number"], workshop=original["workshop"],
                 planted_issues=["invoice_number_reused_from_" + original["claim_id"]],
                 reasons=[f"Invoice {original['invoice_number']} was already used in claim "
                          f"{original['claim_id']}."])
        # the photo is also copied from 'other', so this claim has two planted issues
        c["planted_issues"].append("photo_reused_from_" + other["claim_id"])
        extra.append(c)
        i += 1
    return extra


def main():
    labels = load_labels()
    if CLAIMS.exists():
        shutil.rmtree(CLAIMS)
    CLAIMS.mkdir(parents=True)

    policy_db, claims = [], []
    for i, (scenario, lab) in enumerate(plan_scenarios(labels), start=1):
        claims.append(build_claim(i, scenario, lab, policy_db))
    claims += add_copycat_claims(claims, policy_db, len(claims) + 1)

    # Shuffle claim order so fraud isn't all at the end, then renumber
    random.shuffle(claims)
    id_map = {}
    for n, c in enumerate(claims, start=1):
        id_map[c["claim_id"]] = f"C{n:03d}"
    for c in claims:
        c["claim_id"] = id_map[c["claim_id"]]
        c["planted_issues"] = [_remap(s, id_map) for s in c["planted_issues"]]
        c["reasons"] = [_remap(s, id_map) for s in c["reasons"]]

    # Hold out 10 claims for the final test; never look at these while building
    test_ids = set(random.sample([c["claim_id"] for c in claims], 10))

    for c in claims:
        folder = CLAIMS / c["claim_id"]
        folder.mkdir()
        shutil.copy(PHOTOS / c["photo_source"], folder / "photo_1.jpg")
        write_claim_form(folder / "claim_form.pdf", c)
        write_estimate(folder / "repair_estimate.pdf", c)
        c["split"] = "test" if c["claim_id"] in test_ids else "dev"

    # --- save outputs ---------------------------------------------------------
    with open(DATA / "policy_records.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(policy_db[0].keys()))
        w.writeheader()
        w.writerows(policy_db)

    with open(DATA / "repair_costs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["damage_type", "severity", "min_pkr", "max_pkr"])
        for (d, s), (lo, hi) in REPAIR_COSTS.items():
            w.writerow([d, s, lo, hi])

    truth = {c["claim_id"]: {k: c[k] for k in (
        "split", "scenario", "expected_decision", "planted_issues", "reasons",
        "photo_source", "true_damage", "policy_number", "invoice_number", "amount_claimed")}
        for c in sorted(claims, key=lambda c: c["claim_id"])}
    with open(DATA / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    with open(DATA / "claims_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "split", "scenario", "expected_decision", "claimant",
                    "amount_claimed", "photo_source", "true_damage"])
        for c in sorted(claims, key=lambda c: c["claim_id"]):
            td = c["true_damage"]
            w.writerow([c["claim_id"], c["split"], c["scenario"], c["expected_decision"],
                        c["claimant"], c["amount_claimed"], c["photo_source"],
                        f"{td['severity']} {td['damage_type']} on {td['part']}"])

    # --- report ---------------------------------------------------------------
    from collections import Counter
    print(f"Created {len(claims)} claims in {CLAIMS}/")
    print("Decisions:", dict(Counter(c["expected_decision"] for c in claims)))
    print("Scenarios:", dict(Counter(c["scenario"] for c in claims)))
    print("Split:", dict(Counter(c["split"] for c in claims)))
    print("Also wrote: policy_records.csv, repair_costs.csv, ground_truth.json, claims_summary.csv")


def _remap(text, id_map):
    # replace old claim ids (e.g. C007) with new ones, longest first to avoid clashes
    for old in sorted(id_map, key=len, reverse=True):
        text = text.replace(old, "\0" + id_map[old][1:])
    return text.replace("\0", "C")


if __name__ == "__main__":
    main()
