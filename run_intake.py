"""
Run the intake agent.

  python run_intake.py C002          -> one claim, prints the result
  python run_intake.py --dev         -> all 48 dev claims, saves results and a photo accuracy score
  python run_intake.py --dev --limit 5   -> first 5 dev claims only (cheap test)

Results are saved in outputs/intake/<claim_id>.json
Test claims are never processed here: they are kept for the final evaluation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from intake import run_intake

OUT = Path("outputs/intake")
TRUTH = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))


def save(result):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{result.claim_id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")


def show(result):
    f, e, p = result.claim_form, result.estimate, result.photo
    print(f"\n=== {result.claim_id} ===")
    print(f"Claimant : {f.claimant_name} | Policy {f.policy_number}")
    print(f"Incident : {f.incident_date} at {f.location} | submitted {f.date_submitted}")
    print(f"Claimed  : PKR {f.amount_claimed:,.0f}" if f.amount_claimed is not None else "Claimed  : ?")
    print(f"Estimate : {e.invoice_number} from {e.workshop}, total PKR "
          f"{e.total:,.0f}" if e.total is not None else "Estimate : ?")
    print(f"Photo    : {p.severity} {p.damage_type} on {p.damaged_part}")
    for d in p.all_damages:
        print(f"           - {d.severity} {d.damage_type} on {d.part}")
    print(f"           {p.description}")
    for i in result.issues:
        print(f"  [{i.severity.upper()}] {i.code}: {i.message}")
    if result.missing_fields:
        print(f"  Missing: {', '.join(result.missing_fields)}")
    print(f"Ready for review: {result.ready_for_review}")


SEV_ORDER = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
KEYWORDS = {"dent": "dent", "crushed": "dent", "scratch": "scratch", "crack": "crack",
            "torn": "crack", "lamp": "lamp broken", "light": "lamp broken",
            "headlight": "lamp broken", "glass": "glass shatter", "window": "glass shatter",
            "windshield": "glass shatter", "tire": "tire flat", "tyre": "tire flat"}


def label_types(truth):
    """Main label type plus any types mentioned in the free-text other_damage column."""
    types = {truth["damage_type"]}
    other = (truth.get("other_damage") or "").lower()
    types |= {t for word, t in KEYWORDS.items() if word in other}
    return types


def score(cid, r):
    truth = TRUTH[cid]["true_damage"]
    ai_types = {d.damage_type for d in r.photo.all_damages} | {r.photo.damage_type}
    return {
        "claim_id": cid,
        "label_type": truth["damage_type"], "label_sev": truth["severity"],
        "ai_type": r.photo.damage_type, "ai_sev": r.photo.severity,
        "ai_all": ";".join(sorted(ai_types)),
        # strict: AI main damage == label main damage
        "strict": r.photo.damage_type == truth["damage_type"],
        # found: the labelled main damage appears anywhere in the AI's damage list,
        # or the AI's main damage is one of the damages in the label
        "found": truth["damage_type"] in ai_types or r.photo.damage_type in label_types(truth),
        "sev_exact": r.photo.severity == truth["severity"],
        "sev_within_one": abs(SEV_ORDER[r.photo.severity] - SEV_ORDER[truth["severity"]]) <= 1,
    }


def report(rows, total, seconds):
    n = len(rows)
    pct = lambda k: f"{sum(r[k] for r in rows)}/{n} = {sum(r[k] for r in rows) / n:.0%}"
    print(f"\nProcessed {n}/{total} claims in {seconds:.0f}s")
    print(f"Damage type, strict (main matches main): {pct('strict')}")
    print(f"Damage type, found (label damage seen):  {pct('found')}")
    print(f"Severity, exact:                          {pct('sev_exact')}")
    print(f"Severity, within one level:               {pct('sev_within_one')}")

    confusions = {}
    for r in rows:
        if not r["strict"]:
            key = f"{r['label_type']} -> {r['ai_type']}"
            confusions.setdefault(key, []).append(r["claim_id"])
    if confusions:
        print("\nMost common confusions (label -> AI):")
        for key, cids in sorted(confusions.items(), key=lambda kv: -len(kv[1])):
            print(f"  {key:<28} {len(cids)}  ({', '.join(cids)})")

    import csv
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/intake_photo_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-claim results: outputs/intake_photo_report.csv  |  full outputs: {OUT}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("claim_id", nargs="?", help="e.g. C002")
    ap.add_argument("--dev", action="store_true", help="run all dev claims")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.claim_id:
        cid = args.claim_id.upper()
        if TRUTH.get(cid, {}).get("split", "dev") != "dev":
            sys.exit(f"{cid} is a TEST claim. Keep it for the final evaluation.")
        result = run_intake(cid)
        save(result)
        show(result)
        return

    if not args.dev:
        ap.print_help()
        return

    ids = [cid for cid, t in TRUTH.items() if t["split"] == "dev"][: args.limit]
    rows, start = [], time.time()
    for cid in ids:
        try:
            r = run_intake(cid)
        except Exception as exc:  # keep going if one claim fails
            print(f"{cid}: FAILED ({type(exc).__name__}: {exc})")
            continue
        save(r)
        rows.append(score(cid, r))
        x = rows[-1]
        print(f"{cid}: AI {r.photo.severity} {r.photo.damage_type:<13} | label "
              f"{x['label_sev']} {x['label_type']:<13} | strict {'OK' if x['strict'] else 'X '}"
              f" | found {'OK' if x['found'] else 'X '}")

    if rows:
        report(rows, len(ids), time.time() - start)


if __name__ == "__main__":
    main()
