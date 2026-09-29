"""
Create the SECOND held-out test set (split "test2") from the 38 photos labelled on
2026-09-29 (data/labels_test2.csv, frozen before any model saw them).

  python generate_test2.py                 -> creates test2: C059 ... C078
  python generate_test2.py --set test3     -> creates test3: C079 ... C098 from the 19 photos test2 did not use
  python generate_test2.py [--set test3] --check   -> only re-runs the safety checks

It only ADDS. generate_claims.py deletes and recreates every claim, so it must not be run
again; this script never touches existing claim folders, and it proves that:
  - every existing claim folder is byte-for-byte unchanged,
  - existing policies, ground truth rows and summary rows are unchanged,
  - no dev or test claim gets a new or different fraud signal because of the new claims
    (no shared claimant names, invoice numbers or look-alike photos).

Mix of claims, decided BEFORE generating (20 claims):
  honest 7 | excluded_tyre 3 | inflated_estimate 3 | exaggerated_damage 2 | new_policy 1
  lapsed_policy 1 | third_party_only 1 | reused_photo 1 | duplicate_invoice 1
  - all 3 tyre-only photos become excluded_tyre claims (the known weak spot)
  - both tyre + body photos (car_080, car_088) are honest claims, to check that the
    tyre review does not hold up honest claims whose story is about the body damage
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import generate_claims as g

LABELS2 = g.DATA / "labels_test2.csv"
FREEZE = g.DATA / "labels_test2_freeze.json"

# Each set: decided BEFORE generating. Both add 1 reused_photo + 1 duplicate_invoice claim.
SETS = {
    "test2": {"seed": 2026, "first_id": 59, "tyre": 3, "forced_honest": ["car_080.jpg", "car_088.jpg"],
              "mix": [("honest", 7), ("inflated_estimate", 3), ("exaggerated_damage", 2),
                      ("new_policy", 1), ("lapsed_policy", 1), ("third_party_only", 1)]},
    # test3 (decided 2026-09-29 15:36, before generating): the 19 photos test2 did not use,
    # same frozen system. No tyre-only photos are left, so no tyre claims.
    "test3": {"seed": 2027, "first_id": 79, "tyre": 0, "forced_honest": [],
              "mix": [("honest", 9), ("inflated_estimate", 4), ("exaggerated_damage", 2),
                      ("new_policy", 1), ("lapsed_policy", 1), ("third_party_only", 1)]},
}
SET = "test2"
SEED, FIRST_ID, MIX, FORCED_HONEST = (SETS[SET][k] for k in ("seed", "first_id", "mix", "forced_honest"))
SNAPSHOT = Path(f"outputs/{SET}_safety_snapshot.json")


def use_set(name):
    global SET, SEED, FIRST_ID, MIX, FORCED_HONEST, SNAPSHOT
    SET = name
    SEED, FIRST_ID, MIX, FORCED_HONEST = (SETS[name][k] for k in ("seed", "first_id", "mix", "forced_honest"))
    SNAPSHOT = Path(f"outputs/{name}_safety_snapshot.json")


def folder_digest(folder):
    h = hashlib.sha256()
    for f in sorted(folder.iterdir()):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def load_labels2():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if hashlib.sha256(LABELS2.read_bytes()).hexdigest() != freeze["sha256"]:
        sys.exit("labels_test2.csv does not match its frozen fingerprint. Stopping.")
    with open(LABELS2, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (g.PHOTOS / r["photo"]).exists()]


def fraud_signals_of_existing(history):
    """Fraud signals for every existing claim with a saved intake, using this history."""
    from data_access import load_intake
    from fraud_agent import check_fraud
    truth = json.loads((g.DATA / "ground_truth.json").read_text(encoding="utf-8"))
    out = {}
    for cid, t in truth.items():
        if t["split"] == SET:  # only claims that existed BEFORE this set
            continue
        try:
            r = load_intake(cid)
        except FileNotFoundError:
            continue
        out[cid] = sorted([s.code, s.severity] for s in check_fraud(cid, r, history).signals)
    return out


def plan(labels):
    """Which photo gets which scenario, for the current set. Deterministic (fixed seed)."""
    rnd = random.Random(SEED)
    tyre_only = [r for r in labels if r["damage_type"] == "tire flat" and not r["other_damage"]]
    forced = [r for r in labels if r["photo"] in FORCED_HONEST]
    pool = [r for r in labels if r not in tyre_only and r not in forced]
    rnd.shuffle(pool)
    out = [("excluded_tyre", r) for r in tyre_only]
    assert len(out) == SETS[SET]["tyre"], f"expected {SETS[SET]['tyre']} tyre-only photos, found {len(out)}"
    for scenario, n in MIX:
        if scenario == "honest":
            picks = forced + pool[:n - len(forced)]
            pool = [r for r in pool if r not in picks]
        elif scenario == "exaggerated_damage":
            picks = [r for r in pool if r["severity"] in ("minor", "moderate")][:n]
            pool = [r for r in pool if r not in picks]
        else:
            picks, pool = pool[:n], pool[n:]
        assert len(picks) == n, f"not enough photos for {scenario}"
        out += [(scenario, r) for r in picks]
    assert pool, "no photo left for the duplicate-invoice claim"
    return out, pool  # pool: unused photos (one is used for the duplicate-invoice claim)


def photos_for_set(labels, truth):
    """test2 uses all 38 frozen photos; test3 only the ones no earlier claim has used."""
    if SET == "test2":
        return labels
    used = {t["photo_source"] for t in truth.values()}
    return [r for r in labels if r["photo"] not in used]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--set", choices=sorted(SETS), default="test2")
    args = ap.parse_args()
    use_set(args.set)

    truth_path = g.DATA / "ground_truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if args.check:
        sys.exit(0 if safety_check() else 1)
    if any(t["split"] == SET for t in truth.values()):
        sys.exit(f"The {SET} claims already exist. Use  python generate_test2.py --set {SET} --check")
    if SET != "test2" and not any(t["split"] == "test2" for t in truth.values()):
        sys.exit("Create test2 first (python generate_test2.py).")

    labels = photos_for_set(load_labels2(), truth)
    old_names = {r["claimant"].lower() for r in csv.DictReader(open(g.DATA / "claims_summary.csv", encoding="utf-8"))}
    old_invoices = {t["invoice_number"] for t in truth.values()}
    old_policies = {r["policy_number"] for r in csv.DictReader(open(g.DATA / "policy_records.csv", encoding="utf-8"))}

    # --- snapshot of everything that must NOT change -------------------------------
    from data_access import build_claims_history
    old_history = build_claims_history()
    snapshot = {
        "folders": {p.name: folder_digest(p) for p in sorted(g.CLAIMS.iterdir()) if p.is_dir()},
        "truth": truth,
        "truth_text": truth_path.read_text(encoding="utf-8"),
        "policies": (g.DATA / "policy_records.csv").read_text(encoding="utf-8"),
        "summary": (g.DATA / "claims_summary.csv").read_text(encoding="utf-8"),
        "fraud": fraud_signals_of_existing(old_history),
    }
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snapshot), encoding="utf-8")

    # --- build the new claims with the original generator's logic ------------------
    g.random.seed(SEED)
    used_names = set(old_names)

    def unique_person():
        while True:
            name = g.fake_person()
            if name.lower() not in used_names:
                used_names.add(name.lower())
                return name

    policy_db, claims = [], []
    scenarios, spare = plan(labels)
    i = FIRST_ID
    for scenario, lab in scenarios:
        c = g.build_claim(i, scenario, lab, policy_db)
        claims.append(c)
        i += 1

    honest = [c for c in claims if c["scenario"] == "honest"
              and c["submitted_date"] <= (g.TODAY - timedelta(days=40)).isoformat()]
    g.random.shuffle(honest)
    # reused photo: same photo as an earlier honest claim of this set, different claimant
    orig = honest[0]
    lab = {"photo": orig["photo_source"], **orig["true_damage"]}
    c = g.build_claim(i, "honest", lab, policy_db, after=date.fromisoformat(orig["submitted_date"]))
    c.update(scenario="reused_photo", expected_decision="investigate",
             planted_issues=["photo_reused_from_" + orig["claim_id"]],
             reasons=[f"Photo is identical to the one in claim {orig['claim_id']} from a different claimant."])
    claims.append(c)
    i += 1
    # duplicate invoice: invoice of another earlier honest claim of this set, fresh unused photo
    orig = honest[1]
    lab = dict(spare[0])
    c = g.build_claim(i, "honest", lab, policy_db, after=date.fromisoformat(orig["submitted_date"]))
    c.update(scenario="duplicate_invoice", expected_decision="investigate",
             invoice_number=orig["invoice_number"], workshop=orig["workshop"],
             planted_issues=["invoice_number_reused_from_" + orig["claim_id"]],
             reasons=[f"Invoice {orig['invoice_number']} was already used in claim {orig['claim_id']}."])
    claims.append(c)

    # --- make identities unique against ALL existing claims -------------------------
    for c in claims:
        if c["scenario"] != "duplicate_invoice":
            while c["invoice_number"] in old_invoices:
                c["invoice_number"] = f"INV-{g.random.randint(100000, 999999)}"
            old_invoices.add(c["invoice_number"])
        c["claimant"] = unique_person()
    for p, c in zip(policy_db, claims):
        new_no = f"PMC-2026-{20000 + int(c['claim_id'][1:])}"
        assert new_no not in old_policies
        p["policy_number"] = new_no
        p["holder"] = c["claimant"]
        c["policy_number"] = new_no

    # --- write the new claims (existing files are only appended to) ----------------
    for c in claims:
        c["split"] = SET
        folder = g.CLAIMS / c["claim_id"]
        if folder.exists():
            sys.exit(f"{folder} already exists. Stopping without writing anything else.")
        folder.mkdir()
        shutil.copy(g.PHOTOS / c["photo_source"], folder / "photo_1.jpg")
        g.write_claim_form(folder / "claim_form.pdf", c)
        g.write_estimate(folder / "repair_estimate.pdf", c)

    with open(g.DATA / "policy_records.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(policy_db[0].keys()))
        w.writerows(policy_db)
    for c in claims:
        truth[c["claim_id"]] = {k: c[k] for k in (
            "split", "scenario", "expected_decision", "planted_issues", "reasons",
            "photo_source", "true_damage", "policy_number", "invoice_number", "amount_claimed")}
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    with open(g.DATA / "claims_summary.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for c in claims:
            td = c["true_damage"]
            w.writerow([c["claim_id"], c["split"], c["scenario"], c["expected_decision"], c["claimant"],
                        c["amount_claimed"], c["photo_source"],
                        f"{td['severity']} {td['damage_type']} on {td['part']}"])

    from collections import Counter
    print(f"Created {len(claims)} {SET} claims: {claims[0]['claim_id']} ... {claims[-1]['claim_id']}")
    print("Scenarios:", dict(Counter(c["scenario"] for c in claims)))
    print("Expected decisions:", dict(Counter(c["expected_decision"] for c in claims)))
    if not safety_check():
        rollback([c["claim_id"] for c in claims])
        sys.exit(1)


def rollback(new_ids):
    """Undo everything this script added, so the project is exactly as before."""
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for cid in new_ids:
        folder = g.CLAIMS / cid
        if folder.exists() and cid not in snap["folders"]:
            shutil.rmtree(folder)
    (g.DATA / "ground_truth.json").write_text(snap["truth_text"], encoding="utf-8")
    (g.DATA / "policy_records.csv").write_text(snap["policies"], encoding="utf-8")
    (g.DATA / "claims_summary.csv").write_text(snap["summary"], encoding="utf-8")
    from data_access import build_claims_history
    build_claims_history(force=True)
    print("ROLLED BACK: the new claims were removed and every file is back to how it was.")


def safety_check():
    from data_access import build_claims_history
    import imagehash
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    truth = json.loads((g.DATA / "ground_truth.json").read_text(encoding="utf-8"))
    problems = []

    for name, digest in snap["folders"].items():
        if folder_digest(g.CLAIMS / name) != digest:
            problems.append(f"existing claim folder {name} changed")
    for cid, row in snap["truth"].items():
        if truth.get(cid) != row:
            problems.append(f"ground truth of {cid} changed")
    if not (g.DATA / "policy_records.csv").read_text(encoding="utf-8").startswith(snap["policies"]):
        problems.append("existing policy records changed")
    if not (g.DATA / "claims_summary.csv").read_text(encoding="utf-8").startswith(snap["summary"]):
        problems.append("existing summary rows changed")

    history = build_claims_history(force=True)  # rebuilt: now includes the new claims
    new_ids = [c for c, t in truth.items() if t["split"] == SET]
    old_ids = [c for c in history if c not in new_ids]
    # look-alike photos between a new claim and any older claim (except the planted reuse)
    planted = {cid for cid in new_ids if truth[cid]["scenario"] == "reused_photo"}
    for n in new_ids:
        if n in planted:
            continue
        hn = imagehash.hex_to_hash(history[n]["photo_hash"])
        for o in old_ids + [x for x in new_ids if x != n]:
            if o in planted:
                continue
            if truth.get(o, {}).get("photo_source") == truth[n]["photo_source"]:
                continue
            if hn - imagehash.hex_to_hash(history[o]["photo_hash"]) <= 6:
                problems.append(f"{n} photo looks like {o}'s photo (would trigger photo_reused)")
    now = fraud_signals_of_existing(history)
    for cid, sig in snap["fraud"].items():
        if now.get(cid) != sig:
            problems.append(f"fraud signals of existing claim {cid} changed: {sig} -> {now.get(cid)}")

    if problems:
        print("\nSAFETY CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return False
    print(f"\nSafety check passed: {len(snap['folders'])} existing claim folders unchanged, "
          f"their ground truth unchanged, and fraud signals of all {len(snap['fraud'])} existing "
          f"claims identical. No look-alike photos.")
    return True


if __name__ == "__main__":
    main()
