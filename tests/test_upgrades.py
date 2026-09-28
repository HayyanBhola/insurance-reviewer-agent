"""Tests for the production upgrades: cache, embedding guard, query expansion, metrics.
No AI calls, free. Run with:  pytest -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import llm  # noqa: E402
import policy_index  # noqa: E402
from eval_retrieval import rank_of_first_hit  # noqa: E402
from metrics import cost_usd, percentile  # noqa: E402


class Answer(BaseModel):
    text: str


class CountingLLM:
    calls = 0

    def invoke(self, messages):
        CountingLLM.calls += 1
        return Answer(text="hello")


def _fake_llm(monkeypatch, tmp_path):
    CountingLLM.calls = 0
    monkeypatch.setattr(llm, "CACHE_FILE", tmp_path / "cache.sqlite")
    monkeypatch.setattr(llm, "_candidates", lambda schema, openai_model=None: [CountingLLM()])


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def test_cache_answers_repeat_calls_for_free(monkeypatch, tmp_path):
    _fake_llm(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_CACHE", "1")
    msgs = [HumanMessage("same question")]
    a = llm.structured_call(Answer, msgs)
    b = llm.structured_call(Answer, msgs)
    assert a == b and CountingLLM.calls == 1


def test_cache_misses_on_different_prompt(monkeypatch, tmp_path):
    _fake_llm(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_CACHE", "1")
    llm.structured_call(Answer, [HumanMessage("one")])
    llm.structured_call(Answer, [HumanMessage("two")])
    assert CountingLLM.calls == 2


def test_cache_can_be_turned_off(monkeypatch, tmp_path):
    _fake_llm(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_CACHE", "0")
    llm.structured_call(Answer, [HumanMessage("x")])
    llm.structured_call(Answer, [HumanMessage("x")])
    assert CountingLLM.calls == 2


def test_cache_key_depends_on_model(monkeypatch, tmp_path):
    _fake_llm(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_CACHE", "1")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-nano")
    llm.structured_call(Answer, [HumanMessage("x")])
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    llm.structured_call(Answer, [HumanMessage("x")])
    assert CountingLLM.calls == 2  # a different model must not reuse the other model's answer


# ---------------------------------------------------------------------------
# embedding model guard
# ---------------------------------------------------------------------------
def test_query_embedding_blocked_when_model_differs(monkeypatch, tmp_path):
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps({
        "files": [{"file": "p.pdf", "scanned": False}],
        "chunks": [{"id": "p:p1:c0", "policy_file": "p.pdf", "page": 1, "section": "S",
                    "text": "damage to tyres and tubes"}],
        "vectors": [[0.1, 0.2, 0.3]],
        "embedding_model": "openai:text-embedding-3-small"}))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    ix = policy_index.PolicyIndex(idx)
    assert ix._embed_query("tyres") is None
    assert "mismatch" in ix.mode
    assert ix.search("tyres", k=1)[0]["id"] == "p:p1:c0"  # keyword search still works


# ---------------------------------------------------------------------------
# query expansion and retrieval scoring
# ---------------------------------------------------------------------------
def test_query_expansion_bridges_everyday_words():
    assert "theft" in policy_index.expand_query("Is my car covered if stolen?")
    assert "intoxicating" in policy_index.expand_query("the driver was drunk")
    assert policy_index.expand_query("nothing special here") == "nothing special here"


def test_rank_of_first_hit():
    results = [{"text": "fire and theft"}, {"text": "Damage to tyres\nand tubes"}]
    assert rank_of_first_hit(results, ["tyres and tubes"]) == 2
    assert rank_of_first_hit(results, ["windscreen"]) is None


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_percentile():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([], 95) is None


def test_cost_usd():
    usage = {"gpt-5.4-nano-2026-03-01": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}
    dollars, unpriced = cost_usd(usage)
    assert abs(dollars - 1.45) < 1e-9 and unpriced == []
    assert cost_usd({"unknown-model": {"input_tokens": 5, "output_tokens": 5}})[0] is None


# ---------------------------------------------------------------------------
# header / footer cleaning
# ---------------------------------------------------------------------------
def test_repeated_header_footer_lines_are_detected():
    pages = [(n, f"CIN: L67200MH2000PLC129408\nSection text number {n}\n{n}  PMG/EGIB/PW/ENG/24")
             for n in range(1, 6)]
    bp = policy_index.find_boilerplate(pages)
    assert policy_index._line_key("CIN: L67200MH2000PLC129408") in bp
    assert policy_index._line_key("7  PMG/EGIB/PW/ENG/24") in bp          # page number ignored
    assert policy_index._line_key("Section text number 3") in bp  # same shape on every page
    assert policy_index._is_boilerplate("Page 3 of 24", set())


def test_codes_are_not_headings():
    assert not policy_index._is_heading("IRDAN115RP0017V01200102/A0002V01202122")
    assert not policy_index._is_heading("CIN: L67200MH2000PLC129408")
    assert not policy_index._is_heading("9 PMG/EGIB/PRIVATECARPLAINLANGUAGE/PW/ENG/24")
    assert policy_index._is_heading("SECTION A1B - DAMAGE TO TYRE(S)")


# ---------------------------------------------------------------------------
# ranking fixes found on the real ICICI / Etiqa policies
# ---------------------------------------------------------------------------
def test_plural_questions_are_expanded():
    terms = policy_index.tokenize(policy_index.expand_query("Are flat or burst tyres covered?"))
    assert "tube" in terms                                   # "tyres" must trigger the tyre glossary
    assert "country" in policy_index.tokenize("Which countries are covered?")  # not "countrie"


def test_phrase_bigrams():
    terms = policy_index.tokenize("loss of value", bigrams=True)
    assert "loss_value" in terms


def test_addon_endorsements_are_detected():
    assert policy_index.is_addon({"section": "IMT. 62. ROADSIDE ASSISTANCE", "text": "x"})
    assert policy_index.is_addon({"section": "General", "text": "Endorsement 101: Thailand"})
    assert not policy_index.is_addon({"section": "SECTION I. LOSS OF OR DAMAGE", "text": "x"})


def test_addon_detected_from_text_and_headings():
    assert policy_index.is_addon({"section": "IMT. 65. WORK AWAY", "text": "tyre protect is an add-on"})
    assert policy_index.is_addon({"section": "x", "text": "In consideration of the additional premium ..."})
    assert not policy_index.is_addon({"section": "x", "text": "by burglary housebreaking or theft"})
    assert policy_index._is_heading("Endorsement 101: Extension of Cover to Thailand")
    assert not policy_index._is_heading("Section A1a are deleted and Section B coverage is")


def test_mentions_of_addons_in_core_text_are_not_addons():
    # bug found on the real Etiqa policy: core clauses that merely MENTION an endorsement
    assert not policy_index.is_addon({"section": "EXPLANATORY NOTES",
                                      "text": "You can buy add-on covers such as Endorsement 89."})
    assert not policy_index.is_addon({"section": "Section A", "text": "loss of value (see Endorsement 89)."})
    assert not policy_index._is_heading("Endorsement89.")
    assert not policy_index._is_heading("IMT 7.")


def test_backup_model_answers_are_not_cached_as_the_main_model(monkeypatch, tmp_path):
    """If the main model fails and the backup answers, that answer must NOT be saved under
    the main model's name (otherwise a benchmark could silently mix two models)."""
    import llm
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel

    class S(BaseModel):
        a: int

    class Failing:
        def invoke(self, messages):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    class Working:
        calls = 0

        def invoke(self, messages):
            Working.calls += 1
            return S(a=1)

    monkeypatch.setattr(llm, "CACHE_FILE", tmp_path / "cache.sqlite")
    monkeypatch.setenv("LLM_CACHE", "1")
    monkeypatch.setattr(llm, "_candidates", lambda schema, openai_model=None: [Failing(), Working()])
    msgs = [HumanMessage("same prompt")]
    assert llm.structured_call(S, msgs).a == 1
    llm.structured_call(S, msgs)
    assert Working.calls == 2  # not served from the cache

