"""
Retrieval evaluation: does the policy search find the right clause?

  python eval_retrieval.py            -> uses the index as built (keywords, or hybrid if embedded)
  python eval_retrieval.py --show-misses
  python eval_retrieval.py --ablation     -> measure each search improvement on/off (free)
  python eval_retrieval.py --validation   -> choose the search METHOD on fresh questions
  python eval_retrieval.py --heldout      -> FINAL fair check on questions never used for tuning

Three question sets (like train / validation / test in machine learning):
  data/retrieval_golden.json   26 questions used while improving the search (dev set)
  data/retrieval_validation.json  19 fresh questions, only for CHOOSING between search
      methods (keywords / embeddings / hybrid). The dev set can't referee that fairly,
      because the keyword rules were tuned on it. Only totals are shown.
  data/retrieval_heldout.json  23 NEW questions, never used for tuning (test set).
      Held-out mode prints only the overall score, never which questions failed, so the
      search can't be tuned to them. Each run is logged with a fingerprint of the search
      code, so you can show the search was frozen before the held-out result was seen.

Golden set: data/retrieval_golden.json
Each question says which policy it is about and phrases the correct clause must contain.
A question counts as "found at k" if any of the top-k chunks contains an expected phrase.

Metrics
  recall@1, recall@3, recall@5 : share of questions whose right clause is in the top 1/3/5
  MRR (mean reciprocal rank)   : 1.0 if always ranked first, 0.5 if always second, ...

Keyword mode is completely free. Hybrid mode embeds 26 short questions (a fraction of a cent).
"""

import argparse
import json
import re
import time
from pathlib import Path

from policy_index import PolicyIndex

GOLDEN = Path("data/retrieval_golden.json")
HELDOUT = Path("data/retrieval_heldout.json")
VALIDATION = Path("data/retrieval_validation.json")
HELDOUT_LOG = Path("outputs/heldout_runs.jsonl")


def _norm(text):
    return re.sub(r"\s+", " ", text.lower())


def rank_of_first_hit(results, phrases):
    """1-based rank of the first chunk containing an expected phrase, or None."""
    wanted = [_norm(p) for p in phrases]
    for rank, chunk in enumerate(results, start=1):
        text = _norm(chunk["text"])
        if any(p in text for p in wanted):
            return rank
    return None


def evaluate(index=None, k_max=5, show_misses=False, quiet=False, golden_path=GOLDEN, mode="hybrid"):
    index = index or PolicyIndex()
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    available = index.searchable_files()
    questions = [q for q in golden if q["policy_file"] in available]
    skipped = len(golden) - len(questions)

    rows, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        results = index.search(q["question"], k=k_max, files={q["policy_file"]}, mode=mode)
        latencies.append(time.perf_counter() - t0)
        rows.append({"id": q["id"], "question": q["question"], "file": q["policy_file"],
                     "rank": rank_of_first_hit(results, q["expected_phrases"]),
                     "top": results[0]["section"] if results else ""})

    n = len(rows)
    if n == 0:
        print("No golden questions match the indexed policies. Build the index first.")
        return {}

    def recall(k):
        return sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n

    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    lat = sorted(latencies)
    metrics = {"questions": n, "recall@1": round(recall(1), 3), "recall@3": round(recall(3), 3),
               "recall@5": round(recall(5), 3), "mrr": round(mrr, 3),
               "mode": index.mode if mode == "hybrid" else mode,
               "p95_search_ms": round(lat[int(0.95 * (n - 1))] * 1000, 1)}

    if not quiet:
        print(f"Retrieval evaluation | {n} questions | search mode: {index.mode}"
              + (f" | skipped {skipped} (policy not indexed)" if skipped else ""))
        print(f"  recall@1 = {metrics['recall@1']:.0%}   recall@3 = {metrics['recall@3']:.0%}   "
              f"recall@5 = {metrics['recall@5']:.0%}   MRR = {metrics['mrr']:.2f}   "
              f"p95 search time = {metrics['p95_search_ms']} ms")
        misses = [r for r in rows if not r["rank"] or r["rank"] > 3]
        if show_misses and misses:
            print("\nNot in the top 3:")
            for r in misses:
                print(f"  {r['id']} [{r['file']}] rank={r['rank']} | {r['question']}  "
                      f"(top result: {r['top'][:50]})")
    return metrics


def ablation():
    """Turn each search improvement off in turn and measure the difference.
    Shows which ideas actually help on YOUR policies (a classic ML 'ablation study')."""
    import policy_index as pi
    from rank_bm25 import BM25Okapi

    original = (pi.ADDON_PENALTY, pi.expand_query, pi.tokenize)

    def run(label, penalty=True, bigrams=True, expansion=True):
        pi.ADDON_PENALTY = original[0] if penalty else 1.0
        pi.expand_query = original[1] if expansion else (lambda q: q)
        pi.tokenize = original[2] if bigrams else (lambda t, bigrams=False: original[2](t))
        ix = pi.PolicyIndex()
        if not bigrams:
            ix.bm25 = BM25Okapi([original[2](c["text"]) for c in ix.chunks])
        m = evaluate(ix, quiet=True, mode="keywords")
        print(f"  {label:<30} recall@1 {m['recall@1']:>4.0%}   recall@3 {m['recall@3']:>4.0%}   "
              f"recall@5 {m['recall@5']:>4.0%}   MRR {m['mrr']:.2f}")

    ix0 = pi.PolicyIndex()
    if ix0.vectors:
        print("Search methods (tuning set):")
        for label, mode in [("keywords only", "keywords"), ("embeddings only", "embeddings"),
                            ("hybrid (keywords + embeddings)", "hybrid")]:
            m = evaluate(ix0, quiet=True, mode=mode)
            print(f"  {label:<30} recall@1 {m['recall@1']:>4.0%}   recall@3 {m['recall@3']:>4.0%}   "
                  f"recall@5 {m['recall@5']:>4.0%}   MRR {m['mrr']:.2f}")
        print()
    else:
        print("(No embeddings in the index, so only keyword search is compared.)\n")

    print("Keyword-search improvements:")
    try:
        run("all improvements")
        run("without add-on penalty", penalty=False)
        run("without phrase matching", bigrams=False)
        run("without query expansion", expansion=False)
        run("plain BM25 (none of them)", penalty=False, bigrams=False, expansion=False)
    finally:
        pi.ADDON_PENALTY, pi.expand_query, pi.tokenize = original


def validation():
    """Compare search methods on fresh questions (totals only, no per-question detail)."""
    ix = PolicyIndex()
    print("VALIDATION (fresh questions, used only to choose the search method)")
    modes = [("keywords only", "keywords")]
    if ix.vectors:
        modes += [("embeddings only", "embeddings"), ("hybrid (keywords + embeddings)", "hybrid")]
    else:
        print("  (no embeddings in the index, so only keywords can be measured)")
    for label, mode in modes:
        m = evaluate(ix, quiet=True, golden_path=VALIDATION, mode=mode)
        print(f"  {label:<30} recall@1 {m['recall@1']:>4.0%}   recall@3 {m['recall@3']:>4.0%}   "
              f"recall@5 {m['recall@5']:>4.0%}   MRR {m['mrr']:.2f}   ({m['questions']} questions)")


def search_fingerprint():
    """Short hash of the search code + index, to prove the search didn't change between runs."""
    import hashlib
    from policy_index import INDEX_FILE
    h = hashlib.sha256(Path("policy_index.py").read_bytes())
    h.update(Path(INDEX_FILE).read_bytes())
    return h.hexdigest()[:12]


def heldout():
    import datetime
    fp = search_fingerprint()
    previous = []
    if HELDOUT_LOG.exists():
        previous = [json.loads(line) for line in HELDOUT_LOG.read_text(encoding="utf-8").splitlines() if line]
    print("HELD-OUT EVALUATION (questions never used for tuning; only the total is shown)")
    m = evaluate(quiet=True, golden_path=HELDOUT)
    if not m:
        return
    print(f"  {m['questions']} questions | mode {m['mode']} | recall@1 {m['recall@1']:.0%} | "
          f"recall@3 {m['recall@3']:.0%} | recall@5 {m['recall@5']:.0%} | MRR {m['mrr']:.2f}")
    print(f"  search fingerprint: {fp}")
    if previous and previous[-1]["fingerprint"] != fp:
        print("  ! The search changed since the last held-out run. If that change was made after "
              "seeing the earlier held-out score, this is no longer a clean held-out result.")
    elif previous:
        print(f"  Same search as the previous held-out run ({previous[-1]['time']}).")
    HELDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(HELDOUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                            "fingerprint": fp, **m}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--heldout", action="store_true", help="final check on never-tuned questions")
    ap.add_argument("--validation", action="store_true", help="choose the search method")
    args = ap.parse_args()
    if args.ablation:
        ablation()
    elif args.heldout:
        heldout()
    elif args.validation:
        validation()
    else:
        evaluate(show_misses=args.show_misses)
