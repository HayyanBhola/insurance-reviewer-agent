"""
Why did a golden question miss? (free, no API calls)

  python diagnose_retrieval.py            -> every question not ranked in the top 3
  python diagnose_retrieval.py Q08 Q14    -> specific questions

For each question it reports one of two causes:
  A) PHRASE NOT FOUND - the expected wording doesn't appear anywhere in that policy's text.
     The golden set needs a better phrase (the PDF words it differently or text extraction
     changed it). It prints nearby text for the key word so you can pick the right phrase.
  B) RANKED TOO LOW   - the right clause exists, but search ranks other chunks above it.
     It prints the rank of the right chunk and what beat it.
"""

import json
import re
import sys

from eval_retrieval import GOLDEN, rank_of_first_hit
from policy_index import PolicyIndex, expand_query, is_addon, tokenize

WINDOW = 90


def norm(text):
    return re.sub(r"\s+", " ", text.lower())


def keyword_contexts(chunks, phrases, limit=4):
    """Show text around the most distinctive word of the expected phrases."""
    words = {w for p in phrases for w in re.findall(r"[a-z]+", p.lower()) if len(w) > 3}
    if not words:
        return None, []
    stems = {(w[:-1] if w.endswith("s") else w) for w in words}
    counts = {st: sum(st in norm(c["text"]) for c in chunks) for st in stems}
    present = {st: n for st, n in counts.items() if n > 0}
    # the rarest word that exists is the most useful one to look around
    stem = min(present, key=present.get) if present else max(stems, key=len)
    out = []
    for c in chunks:
        t = norm(c["text"])
        for m in re.finditer(re.escape(stem), t):
            out.append((c["id"], t[max(0, m.start() - WINDOW): m.end() + WINDOW]))
            break
        if len(out) >= limit:
            break
    return stem, out


def main():
    index = PolicyIndex()
    golden = {q["id"]: q for q in json.loads(GOLDEN.read_text(encoding="utf-8"))}
    wanted = [a.upper() for a in sys.argv[1:]]

    for qid, q in golden.items():
        if q["policy_file"] not in index.searchable_files():
            continue
        in_file = [c for c in index.chunks if c["policy_file"] == q["policy_file"]]
        ranked = index.search(q["question"], k=len(in_file), files={q["policy_file"]})
        rank = rank_of_first_hit(ranked, q["expected_phrases"])
        if wanted and qid not in wanted:
            continue
        if not wanted and rank is not None and rank <= 3:
            continue

        print(f"\n=== {qid} [{q['policy_file']}] {q['question']}")
        print(f"    expected phrases: {q['expected_phrases']}")
        holders = [c for c in in_file if any(norm(p) in norm(c["text"]) for p in q["expected_phrases"])]

        if not holders:
            stem, ctx = keyword_contexts(in_file, q["expected_phrases"])
            print(f"    CAUSE A: PHRASE NOT FOUND in this policy's text.")
            if ctx:
                print(f"    Text around '{stem}' (pick a phrase from here for the golden set):")
                for cid, snippet in ctx:
                    print(f"      [{cid}] ...{snippet}...")
            else:
                print(f"    The word '{stem}' does not appear either; this topic may not be in the policy.")
            continue

        print(f"    CAUSE B: RANKED TOO LOW. Right clause is at rank {rank} of {len(ranked)} "
              f"(found in {len(holders)} chunk(s), e.g. {holders[0]['id']}, section '{holders[0]['section'][:50]}').")
        if any(is_addon(h) for h in holders):
            print("    ! The right chunk is flagged as ADD-ON, so its score is reduced. If it is core "
                  "cover, the add-on detection is too broad for this policy.")
        print("    Ranked above it:")
        for i, c in enumerate(ranked[: min(3, (rank or 4) - 1)], start=1):
            tag = " [add-on]" if is_addon(c) else ""
            print(f"      {i}. [{c['id']}]{tag} {c['section'][:45]} | {norm(c['text'])[:110]}...")
        print(f"    Search words used (after expansion): {sorted(set(tokenize(expand_query(q['question']))))}")


if __name__ == "__main__":
    main()
