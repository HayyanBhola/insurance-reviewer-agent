"""
Policy search (the "R" in RAG).

  python policy_index.py build            -> read data/policies/*.pdf, chunk, embed, save index
  python policy_index.py build --no-embed -> keyword search only (free, no API calls)
  python policy_index.py search "damage to tyres"   -> try a query

How it works
------------
1. Each policy PDF is read page by page. Scanned PDFs (pictures of pages, no text)
   are detected and skipped with a warning, because we can't search them.
2. Text is split into chunks of about 900 characters. Each chunk remembers its
   policy file, page number and the nearest section heading, so the coverage agent
   can cite "etiqa_private_car, page 7, Section A1b".
3. Two searches run for every question:
     - BM25 keyword search: great for exact words like "tyre" or "windscreen"
     - embedding search:    great for meaning ("flat tire" ~ "damage to tyres")
   Their rankings are merged with Reciprocal Rank Fusion (hybrid search).
4. The index is saved to data/policy_index.json so embeddings are paid for once.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

from pypdf import PdfReader

POLICY_DIR = Path("data/policies")
INDEX_FILE = Path("data/policy_index.json")
CHUNK_CHARS = 900
OVERLAP_CHARS = 150
MIN_CHARS_PER_PAGE = 200  # less than this on average = probably a scanned PDF

STOPWORDS = set("""a an the of to in on for by with and or is are be been was were this that
these those any all your you our we us it its as at from which shall will may not no if
such other than under into upon their there what how does do did my me i have has get
happens happen much can could would should""".split())

# Query expansion: everyday words people use -> the formal words policies use.
# Free alternative (or complement) to embeddings for closing the vocabulary gap.
QUERY_EXPANSIONS = {
    # keys are normalised words (singular), values use the policies' own wording
    "stolen": "theft burglary housebreaking", "steal": "theft burglary", "stole": "theft burglary",
    "drunk": "intoxicating liquor alcohol influence", "alcohol": "intoxicating liquor",
    "drugs": "intoxicating influence", "accident": "accidental external means collision overturning",
    "crash": "accidental collision", "collision": "accidental collision",
    "injury": "bodily injury death person", "injured": "bodily injury", "hurt": "bodily injury",
    "riot": "riot strike civil commotion", "protest": "riot strike civil commotion",
    "flood": "flood inundation storm", "rain": "flood storm inundation",
    "fire": "fire explosion lightning", "burn": "fire explosion", "burnt": "fire explosion",
    "licence": "driving licence", "license": "driving licence",
    "country": "geographical area", "abroad": "geographical area",
    "outside": "geographical area", "deductible": "deductible excess", "excess": "excess deductible",
    "myself": "excess deductible bear", "windscreen": "windscreen glass windows breakage",
    "windshield": "windscreen glass", "glass": "windscreen glass breakage",
    "value": "depreciation value", "engine": "mechanical electrical breakdown",
    "breakdown": "mechanical electrical breakdown", "tyre": "tyre tube", "burst": "tyre tube",
    "flat": "tyre tube puncture",
}


def expand_query(query):
    words = re.findall(r"[a-z]+", query.lower())
    extra = []
    for w in words:
        key = w if w in QUERY_EXPANSIONS else _normalize(w)  # "tyres" -> "tyre", "countries" -> "country"
        if key in QUERY_EXPANSIONS and QUERY_EXPANSIONS[key] not in extra:
            extra.append(QUERY_EXPANSIONS[key])
    return query + (" " + " ".join(extra) if extra else "")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _line_key(line):
    """Normalise a line so the same header/footer matches on every page
    (page numbers and other digits are replaced by #)."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def find_boilerplate(pages, min_share=0.4):
    """Lines that repeat on many pages (company numbers, document codes, page footers).
    They carry no meaning and confuse both headings and keyword search."""
    if len(pages) < 3:
        return set()
    counts = {}
    for _, text in pages:
        for key in {_line_key(l) for l in text.splitlines() if l.strip()}:
            counts[key] = counts.get(key, 0) + 1
    return {k for k, c in counts.items() if c / len(pages) >= min_share}


def _is_boilerplate(line, boilerplate):
    s = line.strip()
    return (_line_key(s) in boilerplate
            or re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", s, re.I) is not None)


def _is_heading(line):
    s = line.strip()
    if not (3 <= len(s) <= 90):
        return False
    # document codes like "IRDAN115RP0017V01200102" or "CIN: L67200MH..." are not headings
    if " " not in s or sum(c.isdigit() for c in s) / len(s) > 0.2 or s.count("/") >= 2:
        return False
    # add-on headings: "Endorsement 101: Extension of cover ...", "IMT. 24. Electrical fittings"
    # a heading needs a title after the number ("Endorsement 101: Extension of cover");
    # a bare reference like "Endorsement 89." or "see IMT 7." is not a heading
    if re.match(r"^(endorsement|imt\.?)\s*\d+\s*[:.\-\u2013]?\s*[a-z(]{2,}", s, re.I):
        return True
    # "Section A: Loss or damage ..." / "SECTION II - ..." / "Section 3." (not sentences that
    # merely start with the word Section, like "Section A1a are deleted and ...")
    if re.match(r"^(section|part|chapter)\s+[a-z0-9]{1,4}\s*([:\-\u2013.]|$)", s, re.I):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def chunk_pdf(path):
    """Return (chunks, info). Each chunk: id, policy_file, page, section, text."""
    reader = PdfReader(str(path))
    pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    total = sum(len(t.strip()) for _, t in pages)
    info = {"file": path.name, "pages": len(pages), "chars": total,
            "scanned": total / max(len(pages), 1) < MIN_CHARS_PER_PAGE}
    if info["scanned"]:
        return [], info

    boilerplate = find_boilerplate(pages)
    info["boilerplate_lines_removed"] = len(boilerplate)
    stem, chunks, section = path.stem, [], "General"
    for page_no, text in pages:
        buf, n = "", 0
        for line in text.splitlines():
            line = line.strip()
            if not line or _is_boilerplate(line, boilerplate):
                continue
            if _is_heading(line):
                section = line[:90]
            if len(buf) + len(line) > CHUNK_CHARS and buf:
                chunks.append({"id": f"{stem}:p{page_no}:c{n}", "policy_file": path.name,
                               "page": page_no, "section": section, "text": buf.strip()})
                n += 1
                buf = buf[-OVERLAP_CHARS:]  # keep a little overlap for context
            buf += line + "\n"
        if buf.strip():
            chunks.append({"id": f"{stem}:p{page_no}:c{n}", "policy_file": path.name,
                           "page": page_no, "section": section, "text": buf.strip()})
    return chunks, info


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_index(use_embeddings=True):
    pdfs = sorted(POLICY_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {POLICY_DIR}/")

    chunks, files = [], []
    for pdf in pdfs:
        c, info = chunk_pdf(pdf)
        files.append(info)
        if info["scanned"]:
            print(f"  ! {pdf.name}: looks SCANNED ({info['chars']} characters of text in "
                  f"{info['pages']} pages). Skipped - it would need OCR to be searchable.")
        else:
            print(f"  + {pdf.name}: {info['pages']} pages -> {len(c)} chunks, "
                  f"{sum(is_addon(x) for x in c)} of them add-on/endorsement text "
                  f"({info['boilerplate_lines_removed']} repeated header/footer lines removed)")
        chunks += c

    vectors, emb_model = None, None
    if use_embeddings and chunks:
        from llm import embedding_model_name, get_embeddings
        emb = get_embeddings()
        if emb is not None:
            try:
                print(f"  Embedding {len(chunks)} chunks with {embedding_model_name()} "
                      f"(one-time cost, a fraction of a cent)...")
                vectors = emb.embed_documents([c["text"] for c in chunks])
                emb_model = embedding_model_name()  # saved so searches can check they match
            except Exception as exc:
                print(f"  ! Embedding failed ({str(exc)[:150]}). Continuing with keyword search only.")

    INDEX_FILE.write_text(json.dumps({
        "files": files, "chunks": chunks, "vectors": vectors, "embedding_model": emb_model,
    }), encoding="utf-8")
    searchable = [f["file"] for f in files if not f["scanned"]]
    print(f"Saved {INDEX_FILE} | {len(chunks)} chunks | searchable: {searchable} | "
          f"search mode: {'hybrid (keywords + embeddings)' if vectors else 'keywords only'}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
# Policies use British spelling and formal words; claimants use American/everyday words.
# Mapping both to one form lets keyword search match "flat tire" with "damage to tyres".
SYNONYMS = {"tire": "tyre", "windshield": "windscreen", "headlight": "headlamp",
            "taillight": "lamp", "light": "lamp", "car": "vehicle", "auto": "vehicle",
            "fender": "wing", "hood": "bonnet", "trunk": "boot"}


def _normalize(word):
    # very light plural stemming: countries -> country, tyres -> tyre, lamps -> lamp
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return SYNONYMS.get(word, word)


def tokenize(text, bigrams=False):
    """Words -> normalised search terms. With bigrams=True, pairs of neighbouring
    terms are added too ("loss_value", "tyre_tube"), so phrases score higher than
    the same words scattered around a chunk."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    terms = [_normalize(w) for w in words if w not in STOPWORDS and len(w) > 1]
    if bigrams:
        terms += [f"{a}_{b}" for a, b in zip(terms, terms[1:])]
    return terms


# Add-on endorsements (e.g. ICICI "IMT. 24 ...", Etiqa "Endorsement 101 ...") repeat words
# like fire, theft or flat tyre, but only apply if the customer bought them. Our policy
# records hold base cover only, so keyword scores for add-on chunks are reduced.
ADDON_SECTION = re.compile(r"^\s*(imt\.?\s*\d+|endorsement\s*\d+)", re.I)
# markers that only count when a chunk STARTS with them (the chunk IS an add-on),
# not when core text merely mentions one ("see Endorsement 89", "add-on covers are available")
ADDON_START = re.compile(r"^\W*(imt\.?\s*\d+|endorsement\s*\d+|add[- ]?on\s+cover)", re.I)
# the operative wording of an endorsement: counts anywhere in the chunk
ADDON_ANYWHERE = re.compile(r"in consideration of (the|an) additional premium", re.I)
ADDON_PENALTY = 0.6


def is_addon(chunk):
    """Is this chunk add-on / endorsement text? Decided by its own section heading, how the
    chunk starts, or endorsement wording - NOT by a passing mention inside core text."""
    return bool(ADDON_SECTION.search(chunk["section"])
                or ADDON_START.search(chunk["text"][:120])
                or ADDON_ANYWHERE.search(chunk["text"]))


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: merge several ranked lists of ids into one."""
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class PolicyIndex:
    def __init__(self, path=INDEX_FILE):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python policy_index.py build")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.files = data["files"]
        self.chunks = data["chunks"]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.vectors = data.get("vectors")
        self.embedding_model = data.get("embedding_model")
        self._emb = None
        self._warned = False
        self.mode = "hybrid" if self.vectors else "keywords"
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(c["text"], bigrams=True) for c in self.chunks]) if self.chunks else None
        self.addon = [is_addon(c) for c in self.chunks]

    def searchable_files(self):
        return {f["file"] for f in self.files if not f["scanned"]}

    def _embed_query(self, query):
        if self.vectors is None:
            return None
        # Query and documents MUST use the same embedding model, otherwise the
        # similarity numbers are meaningless. If they differ, use keywords only.
        from llm import embedding_model_name
        current = embedding_model_name()
        if current != self.embedding_model:
            if not self._warned:
                print(f"[policy search] Index was embedded with '{self.embedding_model}' but the "
                      f"current embedding model is '{current}'. Using keyword search only. "
                      f"Rebuild with: python policy_index.py build")
                self._warned = True
            self.mode = "keywords (embedding model mismatch)"
            return None
        if self._emb is None:
            from llm import get_embeddings
            self._emb = get_embeddings() or False
        if not self._emb:
            return None
        try:
            return self._emb.embed_query(query)
        except Exception as exc:
            if not self._warned:
                print(f"[policy search] Query embedding failed ({str(exc)[:100]}); keyword search only.")
                self._warned = True
            self.mode = "keywords (query embedding failed)"
            return None

    def search(self, query, k=5, files=None):
        """Hybrid search. files: optional set of policy file names to search within."""
        if not self.chunks:
            return []
        allowed = [i for i, c in enumerate(self.chunks) if not files or c["policy_file"] in files]
        if not allowed:
            return []

        kw_scores = self.bm25.get_scores(tokenize(expand_query(query), bigrams=True))
        kw_scores = [s * ADDON_PENALTY if self.addon[i] else s for i, s in enumerate(kw_scores)]
        kw_rank = [self.chunks[i]["id"] for i in sorted(allowed, key=lambda i: -kw_scores[i])][:30]
        rankings = [kw_rank]

        sims = {}
        qv = self._embed_query(query)
        if qv is not None:
            sims = {self.chunks[i]["id"]: _cosine(qv, self.vectors[i]) for i in allowed}
            rankings.append(sorted(sims, key=sims.get, reverse=True)[:30])

        idx = {c["id"]: i for i, c in enumerate(self.chunks)}
        out = []
        for cid in rrf(rankings)[:k]:
            # return a copy with the retrieval scores attached (useful for tracing/debugging)
            out.append({**self.by_id[cid], "scores": {
                "bm25": round(float(kw_scores[idx[cid]]), 3),
                "cosine": round(sims[cid], 3) if cid in sims else None}})
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--no-embed", action="store_true", help="keyword search only, no API calls")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    if args.cmd == "build":
        build_index(use_embeddings=not args.no_embed)
    else:
        index = PolicyIndex()
        for c in index.search(args.query, k=args.k):
            print(f"\n[{c['id']}] {c['policy_file']} p.{c['page']} | {c['section']} | scores {c['scores']}"
                  f"\n{c['text'][:500]}")
        print(f"\n(search mode: {index.mode})")
