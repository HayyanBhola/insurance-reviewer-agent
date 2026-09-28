"""Embedding path tests with a stand-in embedding model (no API calls, free)."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm  # noqa: E402
import policy_index  # noqa: E402


class FakeEmbeddings:
    """Bag-of-words hashed into 64 dims: similar words -> similar vectors. Counts API calls."""

    def __init__(self, fail_first=0):
        self.doc_calls = 0
        self.query_calls = 0
        self.fail_first = fail_first

    @staticmethod
    def _vec(text):
        v = [0.0] * 64
        for w in policy_index.tokenize(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        return v

    def embed_documents(self, texts):
        self.doc_calls += 1
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED: rate limit")
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vec(text)


def test_batches_and_retries_on_rate_limit():
    emb = FakeEmbeddings(fail_first=2)
    waits = []
    vecs = policy_index.embed_in_batches(emb, [f"text {i}" for i in range(70)], batch=32,
                                         sleep=waits.append)
    assert len(vecs) == 70
    assert emb.doc_calls == 3 + 2      # 3 batches + 2 failed attempts
    assert waits == [5, 10]            # waited, then waited longer


def test_non_rate_limit_errors_are_not_retried():
    class Broken:
        def embed_documents(self, texts):
            raise ValueError("invalid API key")
    try:
        policy_index.embed_in_batches(Broken(), ["a"], sleep=lambda s: None)
        assert False, "should have raised"
    except ValueError:
        pass


def _index_with_vectors(tmp_path, monkeypatch, emb):
    chunks = [
        {"id": "p:p1:c0", "policy_file": "p.pdf", "page": 1, "section": "S",
         "text": "loss or damage by malicious act"},
        {"id": "p:p1:c1", "policy_file": "p.pdf", "page": 1, "section": "S",
         "text": "damage to tyres and tubes unless the vehicle is damaged"},
    ]
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"files": [{"file": "p.pdf", "scanned": False}], "chunks": chunks,
                                "vectors": [FakeEmbeddings._vec(c["text"]) for c in chunks],
                                "embedding_model": "google_genai:test-embedding"}))
    monkeypatch.setenv("EMBEDDING_MODEL", "google_genai:test-embedding")
    monkeypatch.setattr(policy_index, "QUERY_CACHE_FILE", tmp_path / "qcache.json")
    monkeypatch.setattr(llm, "get_embeddings", lambda: emb)
    return policy_index.PolicyIndex(path)


def test_query_embeddings_are_cached(tmp_path, monkeypatch):
    emb = FakeEmbeddings()
    ix = _index_with_vectors(tmp_path, monkeypatch, emb)
    ix.search("tyres covered?", k=1)
    ix.search("tyres covered?", k=1)
    assert emb.query_calls == 1
    # a fresh index object (e.g. the next run) reads the cache from disk
    ix2 = policy_index.PolicyIndex(tmp_path / "index.json")
    ix2._emb = emb
    ix2.search("tyres covered?", k=1)
    assert emb.query_calls == 1


def test_search_modes(tmp_path, monkeypatch):
    emb = FakeEmbeddings()
    ix = _index_with_vectors(tmp_path, monkeypatch, emb)
    assert ix.search("tyres", k=1, mode="keywords")[0]["scores"]["cosine"] is None
    assert ix.search("tyres", k=1, mode="embeddings")[0]["scores"]["cosine"] is not None
    assert ix.search("tyres", k=1, mode="hybrid")[0]["id"] == "p:p1:c1"


def test_hybrid_uses_embeddings_when_no_keyword_matches(tmp_path, monkeypatch):
    ix = _index_with_vectors(tmp_path, monkeypatch, FakeEmbeddings())
    # "tubes" matches nothing useful for BM25 here (2-doc corpus gives zero scores),
    # so the embedding ranking alone must decide
    assert ix.search("tyres", k=1, mode="hybrid")[0]["id"] == "p:p1:c1"


def test_overlap_starts_at_line_boundary():
    buf = "first line of the chunk\nsecond line about damage to tyres and tubes unless\n"
    out = policy_index._overlap(buf, limit=60)
    assert out.startswith("second line")          # whole line, not a word fragment
    long = "word " * 100
    assert not policy_index._overlap(long, limit=30).startswith("ord")


def test_addon_rule_applies_to_embedding_ranking(tmp_path, monkeypatch):
    ix = _index_with_vectors(tmp_path, monkeypatch, FakeEmbeddings())
    ix.addon = [False, True]                        # chunk c1 is an add-on
    monkeypatch.setattr(ix, "_embed_query", lambda q: [1.0, 0.0])
    # close call: add-on cosine 0.9 vs core 0.7 -> 0.9 * 0.6 = 0.54 < 0.7, core wins
    ix.vectors = [[0.7, 0.714], [0.9, 0.436]]
    assert ix.search("q", k=2, mode="embeddings")[0]["id"] == "p:p1:c0"
    # clear gap: add-on 0.99 vs core 0.3 -> 0.59 > 0.3, the soft rule lets the add-on win
    ix.vectors = [[0.3, 0.954], [0.99, 0.141]]
    assert ix.search("q", k=2, mode="embeddings")[0]["id"] == "p:p1:c1"
