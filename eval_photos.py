"""
Photo recognition benchmark: measure the vision step ON ITS OWN, before any claim logic.

  python eval_photos.py                          -> baseline from your saved intake results (FREE)
  python eval_photos.py --run v3 --split tune    -> one prompt on the tuning photos (every mistake shown)
  python eval_photos.py --compare v1 v3          -> tuning photos, side by side (advisory only)
  python eval_photos.py --compare v1 v3 --final  -> THE DECISION: fresh photos, rule decided in advance
  add --fresh to ask the model again instead of using cached answers

Prompt versions (intake.py): v1 original | v2 severity scale + evidence (tested, rejected)
                             v3 = v1 + "pre-existing wear is not damage"

How we avoid overfitting
------------------------
Photo groups, fixed forever:
  tune  (21): dev photos. Every mistake is shown; this is where you look and think.
  check (21): the other dev photos. Totals only.
  extra  (6): labelled tyre photos that NO claim uses. Never looked at. Totals only.
The 3 dev photos also used by TEST claims are left out, and the 10 test claims stay untouched.

The final decision (--final) uses check + extra together and applies KEEP_RULE below,
which was written BEFORE any v3 result existed. Every final run is logged with a
fingerprint of both prompts; a second final run after changing a prompt is flagged.

Metrics
-------
  type accuracy   : main damage type matches the label
  severity exact  : main severity matches the label (within one: off by at most one level)
  over-read       : AI says MORE severe than the label  -> hides inflated bills (C012)
  under-read      : AI says LESS severe than the label  -> honest bills look inflated
  damage found    : the labelled main damage type appears ANYWHERE in the AI's list
  body damage kept: photos whose label has body damage (not just a tyre) where the AI
                    still lists body damage (guard: never hide real damage; a flat tyre
                    WITH body damage is covered, so hiding it would wrongly deny claims)
  tyre-only right : photos labelled "flat tyre, nothing else" that the AI reads as tyre-only
                    (this decides whether the tyre exclusion can apply)
  extra damages   : damages listed beyond what the label has
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("google_genai").setLevel(logging.ERROR)
os.environ["LLM_FALLBACK"] = "0"  # a benchmark must use ONE model: never mix in the backup

TRUTH = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
LABELS = {r["photo"]: r for r in csv.DictReader(open("data/labels.csv", encoding="utf-8"))}
PHOTOS = Path("data/photos")
OUT = Path("outputs/photo_eval")
CHECK_LOG = Path("outputs/photo_check_runs.jsonl")
SEV = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
NOISE = 0.05  # about one photo: smaller differences are run-to-run noise (measured: mini ran twice)

# Decided BEFORE seeing any v3 result. The new prompt is kept only if ALL are true on fresh photos.
KEEP_RULE = [
    ("tyre_only", "more tyre-only photos read correctly", lambda new, old: new > old),
    ("damage_found", "main damage still found", lambda new, old: new >= old - NOISE),
    ("body_kept", "does not hide real body damage", lambda new, old: new >= old - NOISE),
    ("type_accuracy", "damage type not worse (beyond noise)", lambda new, old: new >= old - NOISE),
    ("within_one", "severity not worse (beyond noise)", lambda new, old: new >= old - NOISE),
]


# ---------------------------------------------------------------------------
# photo groups
# ---------------------------------------------------------------------------
def photo_splits():
    """Fixed, disjoint photo groups (by a hash of the file name, so they never change)."""
    dev = {t["photo_source"] for t in TRUTH.values() if t["split"] == "dev"}
    test = {t["photo_source"] for t in TRUTH.values() if t["split"] == "test"}
    used = {t["photo_source"] for t in TRUTH.values()}
    usable = sorted(p for p in dev - test if p in LABELS)
    ranked = sorted(usable, key=lambda p: hashlib.sha1(p.encode()).hexdigest())
    half = len(ranked) // 2
    tune, check = sorted(ranked[:half]), sorted(ranked[half:])
    extra = sorted(p for p in LABELS if p not in used)
    return {"tune": tune, "check": check, "extra": extra, "final": check + extra, "all": usable}


def is_tyre_only(lab):
    return lab["damage_type"] == "tire flat" and not (lab.get("other_damage") or "").strip()


def has_body_damage(lab):
    """True if the label includes any damage that is not a tyre (main or 'other damage')."""
    if lab["damage_type"] != "tire flat":
        return True
    others = [o.strip().lower() for o in (lab.get("other_damage") or "").split(";") if o.strip()]
    return any("tire" not in o and "tyre" not in o for o in others)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score(photo, pred):
    """pred: a PhotoAssessment (or a dict with the same fields)."""
    p = pred if isinstance(pred, dict) else pred.model_dump()
    lab = LABELS[photo]
    diff = SEV.get(p["severity"], 0) - SEV[lab["severity"]]
    ai_types = {d["damage_type"] for d in p["all_damages"]} | {p["damage_type"]}
    ai_types.discard("none")
    n_label_items = 1 + (1 if (lab.get("other_damage") or "").strip() else 0)
    return {
        "photo": photo,
        "label": f"{lab['severity']} {lab['damage_type']} on {lab['part']}"
                 + (f" (+ {lab['other_damage']})" if lab.get("other_damage") else ""),
        "ai": f"{p['severity']} {p['damage_type']} on {p.get('damaged_part')}",
        "ai_all": "; ".join(f"{d['severity']} {d['damage_type']} on {d['part']}" for d in p["all_damages"]),
        "type_ok": p["damage_type"] == lab["damage_type"], "sev_ok": diff == 0,
        "within_one": abs(diff) <= 1, "over": diff > 0, "under": diff < 0,
        "found": lab["damage_type"] in ai_types,
        "tyre_only_label": is_tyre_only(lab),
        "body_label": has_body_damage(lab),
        "body_kept": has_body_damage(lab) and bool(ai_types - {"tire flat"}),
        "tyre_only_ok": is_tyre_only(lab) and bool(ai_types) and ai_types <= {"tire flat"},
        "extra": max(0, len(p["all_damages"]) - n_label_items),
        "sev_pair": (lab["severity"], p["severity"]),
    }


def metrics(rows):
    n = len(rows)
    if not n:
        return {}
    tyre = [r for r in rows if r["tyre_only_label"]]
    body = [r for r in rows if r["body_label"]]

    def share(key, subset=None):
        s = rows if subset is None else subset
        return round(sum(r[key] for r in s) / len(s), 3) if s else None

    return {"photos": n, "type_accuracy": share("type_ok"), "severity_exact": share("sev_ok"),
            "within_one": share("within_one"), "over_read": share("over"), "under_read": share("under"),
            "damage_found": share("found"), "tyre_photos": len(tyre),
            "tyre_only": share("tyre_only_ok", tyre),
            "body_photos": len(body), "body_kept": share("body_kept", body),
            "extra_per_photo": round(sum(r["extra"] for r in rows) / n, 2)}


def _pct(v):
    return "  n/a" if v is None else f"{v:>5.0%}"


def report(rows, title, details):
    m = metrics(rows)
    if not m:
        print("No photos scored.")
        return m
    print(f"\n{title}  ({m['photos']} photos)")
    print(f"  damage type accuracy : {_pct(m['type_accuracy'])}")
    print(f"  severity exact       : {_pct(m['severity_exact'])}   (within one level: {_pct(m['within_one']).strip()})")
    print(f"  severity OVER-read   : {_pct(m['over_read'])}   <- hides inflated bills")
    print(f"  severity UNDER-read  : {_pct(m['under_read'])}   <- makes honest bills look inflated")
    print(f"  damage found in list : {_pct(m['damage_found'])}   <- must not drop (never hide real damage)")
    print(f"  tyre-only read right : {_pct(m['tyre_only'])}   ({m['tyre_photos']} tyre-only photos)")
    print(f"  body damage kept     : {_pct(m['body_kept'])}   ({m['body_photos']} photos with body damage; must not drop)")
    print(f"  extra damages/photo  : {m['extra_per_photo']:>5}")
    if details:
        pairs = Counter(r["sev_pair"] for r in rows)
        order = ["minor", "moderate", "severe"]
        print("\n  Severity: rows = label, columns = AI")
        print("              " + "".join(f"{o:>10}" for o in order))
        for lab in order:
            print(f"  {lab:<12}" + "".join(f"{pairs.get((lab, ai), 0):>10}" for ai in order))
        wrong = [r for r in rows if not (r["type_ok"] and r["sev_ok"])
                 or (r["tyre_only_label"] and not r["tyre_only_ok"]) or (r["body_label"] and not r["body_kept"])]
        if wrong:
            print("\n  Mistakes (tune photos only):")
            for r in wrong:
                print(f"   {r['photo']}: label {r['label']}\n{'':<16}AI    {r['ai']}  | all: {r['ai_all']}")
    return m


# ---------------------------------------------------------------------------
# calling the model (paced, retried, cached)
# ---------------------------------------------------------------------------
def vision_model_name():
    from llm import model_id
    return model_id(os.getenv("OPENAI_VISION_MODEL"))


def collect(version, photos, fresh=False, label=""):
    """Assess each photo with prompt `version`. Returns (rows, predictions, complete?)."""
    from intake import assess_photo, photo_request
    from llm import CACHE_STATS, forget, out_of_credit

    vision = vision_model_name()
    print(f"Prompt {version}{label} on {len(photos)} photos with {vision} ...")
    if fresh:
        gone = sum(forget(*photo_request(PHOTOS / ph, version), os.getenv("OPENAI_VISION_MODEL"))
                   for ph in photos)
        print(f"  --fresh: removed {gone} cached answer(s), asking the model again")
    # Free tiers allow only a few requests per minute: pace new calls, and on a
    # "429 RESOURCE_EXHAUSTED" wait a minute and try again. Finished photos are cached,
    # so if you stop (Ctrl+C) or it gives up, run the same command again.
    delay = float(os.getenv("PHOTO_DELAY", "7" if vision.startswith("google") else "0"))
    rows, preds, limited, t0 = [], {}, False, time.time()
    hits0, misses0 = CACHE_STATS["hits"], CACHE_STATS["misses"]
    for i, ph in enumerate(photos, start=1):
        pred, before = None, CACHE_STATS["misses"]
        for attempt in range(4):
            try:
                pred = assess_photo(PHOTOS / ph, version=version)
                limited = False
                break
            except Exception as exc:  # noqa: BLE001
                if out_of_credit(exc):
                    raise SystemExit(f"\nSTOPPED at {ph}: your API credit/quota is used up "
                                     "(insufficient_quota). Waiting won't help: add credit or switch "
                                     "provider. Finished photos are cached.")
                limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if limited and attempt < 3:
                    print(f"  {ph}: rate limit, waiting 65s (attempt {attempt + 1}/3) ...")
                    time.sleep(65)
                    continue
                print(f"  {ph}: FAILED ({type(exc).__name__}: {str(exc)[:120]})")
                break
        if pred is None:
            if limited:
                print("\nStill rate-limited: the limit may be used up for now. "
                      "Run the same command later; finished photos are cached.")
                break
            continue
        preds[ph] = pred.model_dump()
        rows.append(score(ph, pred))
        new_call = CACHE_STATS["misses"] > before
        print(f"  [{i}/{len(photos)}] {ph} {'(new call)' if new_call else '(cached)'}")
        if new_call and delay:
            time.sleep(delay)
    complete = len(rows) == len(photos)
    print(f"  done in {time.time() - t0:.0f}s | cached {CACHE_STATS['hits'] - hits0}, "
          f"new calls {CACHE_STATS['misses'] - misses0}")
    if not complete:
        print(f"\n!! INCOMPLETE: {len(rows)}/{len(photos)} photos. Scores are NOT comparable yet. "
              "Run the same command again to finish (finished photos are free).")
    return rows, preds, complete


def prompt_fingerprint(version):
    import intake
    text = {"v1": intake.PHOTO_SYSTEM, "v2": intake.PHOTO_SYSTEM_V2, "v3": intake.PHOTO_SYSTEM_V3}[version]
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def _save(version, split, preds):
    OUT.mkdir(parents=True, exist_ok=True)
    safe = vision_model_name().replace(":", "_").replace("/", "_")
    (OUT / f"{version}_{safe}_{split}.json").write_text(json.dumps(preds, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def baseline():
    """Score the photo readings already saved by run_intake.py (tune photos): free."""
    photos = set(photo_splits()["tune"])
    rows, seen = [], set()
    for cid, t in TRUTH.items():
        ph = t["photo_source"]
        f = Path(f"outputs/intake/{cid}.json")
        if t["split"] != "dev" or ph not in photos or ph in seen or not f.exists():
            continue
        seen.add(ph)
        rows.append(score(ph, json.loads(f.read_text(encoding="utf-8"))["photo"]))
    report(rows, "BASELINE: your saved intake readings, tune photos", details=True)


def run(version, split, fresh=False):
    if split != "tune":
        raise SystemExit("Single runs are for the tune photos. For the decision use: --compare v1 v3 --final")
    rows, preds, complete = collect(version, photo_splits()["tune"], fresh)
    report(rows, f"PROMPT {version} | {vision_model_name()} | tune photos", details=True)
    if complete:
        _save(version, "tune", preds)


def decide(old_m, new_m):
    """Apply KEEP_RULE. Returns (keep?, list of (description, passed, old, new))."""
    checks = []
    for key, text, test in KEEP_RULE:
        old, new = old_m.get(key), new_m.get(key)
        passed = old is not None and new is not None and test(new, old)
        checks.append((text, passed, old, new))
    return all(p for _, p, _, _ in checks), checks


def compare(old_v, new_v, final=False, fresh=False):
    split = "final" if final else "tune"
    photos = photo_splits()[split]
    where = "FRESH photos (check + extra), totals only" if final else "tune photos (advisory, not the decision)"
    results = {}
    for v in (old_v, new_v):
        rows, _, complete = collect(v, photos, fresh, label=f" [{split}]")
        if not complete:
            raise SystemExit("Stopped: a run was incomplete, so no comparison was made. Run again.")
        results[v] = metrics(rows)

    old_m, new_m = results[old_v], results[new_v]
    print(f"\n{old_v} vs {new_v} | {vision_model_name()} | {where}")
    print(f"  {'':<24}{old_v:>8}{new_v:>8}")
    for key, name in [("type_accuracy", "damage type"), ("severity_exact", "severity exact"),
                      ("within_one", "severity within one"), ("over_read", "over-read"),
                      ("under_read", "under-read"), ("damage_found", "damage found in list"),
                      ("tyre_only", f"tyre-only read right ({old_m['tyre_photos']})"),
                      ("body_kept", f"body damage kept ({old_m['body_photos']})")]:
        print(f"  {name:<24}{_pct(old_m[key]):>8}{_pct(new_m[key]):>8}")

    keep, checks = decide(old_m, new_m)
    print("\nRule decided in advance (all must pass):")
    for text, passed, _, _ in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {text}")
    verdict = f"KEEP {new_v}" if keep else f"REJECT {new_v} (keep {old_v})"
    print(f"\nVERDICT: {verdict}" + ("" if final else "   <- tune photos only; the real decision is --final"))

    if final:
        fps = {v: prompt_fingerprint(v) for v in (old_v, new_v)}
        previous = [json.loads(x) for x in CHECK_LOG.read_text(encoding="utf-8").splitlines() if x] \
            if CHECK_LOG.exists() else []
        if any(p.get("versions") == [old_v, new_v] for p in previous):
            print("  ! This comparison was already run on the fresh photos. If a prompt changed after "
                  "seeing that result, this is no longer a clean decision.")
        CHECK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                "versions": [old_v, new_v], "model": vision_model_name(),
                                "fingerprints": fps, "verdict": verdict,
                                old_v: old_m, new_v: new_m}) + "\n")
        print(f"  logged to {CHECK_LOG}")
        if keep:
            print(f"\nNext: set PHOTO_PROMPT={new_v} in .env, then run: python refresh_photos.py")
    return keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["v1", "v2", "v3"], help="one prompt on the tune photos")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"), help="e.g. --compare v1 v3")
    ap.add_argument("--final", action="store_true", help="with --compare: the decision on fresh photos")
    ap.add_argument("--split", default="tune", help="(kept for old commands; only 'tune' is allowed)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached answers for these photos")
    args = ap.parse_args()
    if args.compare:
        bad = [v for v in args.compare if v not in ("v1", "v2", "v3")]
        if bad:
            raise SystemExit(f"Unknown prompt version(s): {bad}")
        compare(*args.compare, final=args.final, fresh=args.fresh)
    elif args.run:
        run(args.run, args.split, fresh=args.fresh)
    else:
        baseline()
