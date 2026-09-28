"""
Re-read ONLY the damage photos of your saved intake results with another photo prompt.

  python refresh_photos.py            -> uses PHOTO_PROMPT from .env (e.g. v3)
  python refresh_photos.py --dry-run  -> show what would change, save nothing
  python refresh_photos.py --restore  -> undo: put back the most recent backup

Safe by design
  - The claim form and repair estimate are NOT read again (no cost, no new extraction errors).
  - Validation is re-run, because some checks depend on the photo.
  - The whole outputs/intake folder is backed up first to outputs/intake_backups/<time>/.
  - Only dev claims are touched; test claims stay untouched for the final evaluation.
  - outputs/intake/_photo_prompt.json records which prompt and model produced the photos.
"""

import argparse
import json
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from intake import CLAIMS_DIR, PHOTO_VERSIONS, assess_photo, validate  # noqa: E402
from llm import CACHE_STATS, model_id, out_of_credit  # noqa: E402
from schemas import IntakeResult  # noqa: E402

INTAKE = Path("outputs/intake")
BACKUPS = Path("outputs/intake_backups")
MARKER = INTAKE / "_photo_prompt.json"
TRUTH = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))


def rules_decision(result, history, index):
    """The free, rules-only decision for one intake result (same logic as the graph)."""
    from coverage_agent import check_coverage
    from decision_agent import recommend
    from evidence_agent import check_evidence
    from fraud_agent import check_fraud
    findings = {"coverage": check_coverage(result, index, use_llm=False).model_dump(),
                "evidence": check_evidence(result, use_llm=False).model_dump(),
                "fraud": check_fraud(result.claim_id, result, history).model_dump()}
    return recommend(findings)


STRICTNESS = {"approve": 0, "investigate": 1, "deny": 2}


def _judge(before, after, expected):
    """'better' = now correct; 'safer' = moved toward the right answer (e.g. approve -> investigate
    on a claim that should be denied); 'WORSE' = was right before, or moved away from it."""
    if after == expected:
        return "better"
    if before == expected:
        return "WORSE"
    gap_before = abs(STRICTNESS[before] - STRICTNESS[expected])
    gap_after = abs(STRICTNESS[after] - STRICTNESS[expected])
    return "safer" if gap_after < gap_before else "WORSE"


def show_decision_changes(pairs):
    """pairs: [(old IntakeResult, new IntakeResult)]. Prints rules-only decisions before/after."""
    from data_access import build_claims_history
    from policy_index import INDEX_FILE, PolicyIndex
    history = build_claims_history()
    index = PolicyIndex() if INDEX_FILE.exists() else None
    before_ok = after_ok = 0
    moves = []
    for old, new in pairs:
        t = TRUTH[old.claim_id]
        a, b = rules_decision(old, history, index), rules_decision(new, history, index)
        before_ok += a == t["expected_decision"]
        after_ok += b == t["expected_decision"]
        if a != b:
            better = _judge(a, b, t["expected_decision"])
            moves.append(f"  {old.claim_id} {t['scenario']:<19} expected {t['expected_decision']:<11} "
                         f"{a:>11} -> {b:<11} {better}")
    print(f"\nRules-only decisions: {before_ok}/{len(pairs)} correct before -> {after_ok}/{len(pairs)} after")
    print("\n".join(moves) if moves else "  (no decision changed)")
    print("  Note: rules alone never DENY a tyre claim; 'approve -> investigate' is the right move there."
          " Denying with a policy quote needs the AI coverage check (--llm).")


def _summary(photo):
    return f"{photo.severity} {photo.damage_type}" + (
        f" (+{len(photo.all_damages) - 1} more)" if len(photo.all_damages) > 1 else "")


def restore():
    if not BACKUPS.exists() or not any(BACKUPS.iterdir()):
        raise SystemExit("No backup found.")
    latest = sorted(p for p in BACKUPS.iterdir() if p.is_dir())[-1]
    shutil.rmtree(INTAKE)
    shutil.copytree(latest, INTAKE)
    print(f"Restored outputs/intake from {latest}")


def refresh(version, dry_run):
    if version not in PHOTO_VERSIONS:
        raise SystemExit(f"PHOTO_PROMPT must be one of {PHOTO_VERSIONS}, got {version!r}")
    files = sorted(f for f in INTAKE.glob("C*.json"))
    dev = [f for f in files if TRUTH.get(f.stem, {}).get("split") == "dev"]
    skipped = len(files) - len(dev)
    vision = model_id(os.getenv("OPENAI_VISION_MODEL"))
    print(f"Re-reading {len(dev)} dev claim photos with prompt {version} on {vision}"
          + (f" (skipping {skipped} non-dev)" if skipped else "") + (" [DRY RUN]" if dry_run else ""))

    if not dry_run:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copytree(INTAKE, BACKUPS / stamp)
        print(f"Backup saved to {BACKUPS / stamp}  (undo with: python refresh_photos.py --restore)")

    delay = float(os.getenv("PHOTO_DELAY", "7" if vision.startswith("google") else "0"))
    changed, failed, pairs = [], [], []
    for f in dev:
        old = IntakeResult.model_validate_json(f.read_text(encoding="utf-8"))
        photo = None
        for attempt in range(4):
            before = CACHE_STATS["misses"]
            try:
                photo = assess_photo(CLAIMS_DIR / old.claim_id / "photo_1.jpg", version=version)
                break
            except Exception as exc:  # noqa: BLE001
                if out_of_credit(exc):
                    raise SystemExit(f"\nSTOPPED at {old.claim_id}: API credit/quota used up. Claims done so far "
                                     "are saved; undo everything with: python refresh_photos.py --restore")
                if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and attempt < 3:
                    print(f"  {old.claim_id}: rate limit, waiting 65s ...")
                    time.sleep(65)
                    continue
                print(f"  {old.claim_id}: FAILED ({type(exc).__name__}: {str(exc)[:100]}) - kept the old reading")
                failed.append(old.claim_id)
                break
        if photo is None:
            continue
        issues, missing = validate(old.claim_form, old.estimate, photo)
        new = old.model_copy(update={
            "photo": photo, "issues": issues, "missing_fields": missing,
            "ready_for_review": not missing and not any(i.severity == "error" for i in issues)})
        if _summary(old.photo) != _summary(photo) or {d.damage_type for d in old.photo.all_damages} != \
                {d.damage_type for d in photo.all_damages}:
            changed.append((old.claim_id, _summary(old.photo), _summary(photo)))
        pairs.append((old, new))
        if not dry_run:
            f.write_text(new.model_dump_json(indent=1), encoding="utf-8")
        if CACHE_STATS["misses"] > before and delay:
            time.sleep(delay)

    print(f"\n{len(changed)} photo reading(s) changed:")
    for cid, a, b in changed:
        print(f"  {cid}: {a:<28} -> {b}")
    print(f"cached {CACHE_STATS['hits']}, new calls {CACHE_STATS['misses']}"
          + (f" | {len(failed)} FAILED (old reading kept): {failed}" if failed else ""))
    if pairs:
        show_decision_changes(pairs)
    if not dry_run:
        MARKER.write_text(json.dumps({"photo_prompt": version, "vision_model": vision,
                                      "time": datetime.now().isoformat(timespec="seconds"),
                                      "failed": failed}, indent=1), encoding="utf-8")
        if failed:
            print("Some photos failed: run the same command again to finish them.")
        print("\nIf any claim got WORSE and you don't want that: python refresh_photos.py --restore")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", help="photo prompt (default: PHOTO_PROMPT from .env)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    if args.restore:
        restore()
    else:
        refresh(args.version or os.getenv("PHOTO_PROMPT", "v1"), args.dry_run)
