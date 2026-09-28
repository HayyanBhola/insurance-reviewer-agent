"""
Phase 3 schemas: what each specialist agent returns.

Coverage agent  -> CoverageFinding   (is this claim covered by the policy?)
Evidence agent  -> EvidenceFinding   (does the photo support the claim and the price?)
Fraud agent     -> FraudFinding      (are there fraud warning signs?)
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Flag(BaseModel):
    """One warning raised by an agent."""
    code: str                                   # machine-readable, e.g. "photo_reused"
    severity: Literal["low", "medium", "high"]  # low = FYI, medium/high = needs a human
    message: str                                # human-readable explanation


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
class ClauseCitation(BaseModel):
    """A clause the LLM relied on. The quote must be copied from the retrieved text."""
    chunk_id: str = Field(description="The id of the clause, exactly as given, e.g. 'icici_private_car:p3:c2'")
    quote: str = Field(description="The exact sentence(s) copied word for word from that clause")


class CoverageJudgment(BaseModel):
    """What the LLM decides after reading the retrieved policy clauses."""
    decision: Literal["covered", "not_covered", "needs_review"] = Field(
        description="covered = the policy pays for this damage; not_covered = an exclusion clearly "
                    "applies; needs_review = the clauses are unclear or missing")
    reasoning: str = Field(description="Two or three sentences explaining the decision")
    citations: list[ClauseCitation] = Field(
        description="The one to three clauses that support the decision, with exact quotes")


class VerifiedCitation(BaseModel):
    chunk_id: str
    policy_file: str
    page: int
    section: str
    quote: str
    verified: bool  # True if the quote really appears in that clause (anti-hallucination check)


class CoverageFinding(BaseModel):
    status: Literal["covered", "not_covered", "needs_review"]
    policy_found: bool
    policy_active_on_incident: Optional[bool]
    coverage_type: Optional[str]
    deductible: Optional[float]
    sum_insured: Optional[float]
    policy_wording: Optional[str]
    wording_searchable: bool  # False if the wording PDF is scanned / not indexed
    reasons: list[str]
    citations: list[VerifiedCitation]
    method: Literal["rules", "rules+rag"]  # did we need the LLM?


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
class DescriptionCheck(BaseModel):
    """LLM comparison of what the claimant wrote vs what the photo shows."""
    consistent: bool = Field(
        description="False if the description claims clearly MORE or DIFFERENT damage than the photo shows")
    claimed_damage_summary: str = Field(description="Short summary of the damage the claimant describes")
    explanation: str = Field(description="One or two sentences explaining the judgement")


class EvidenceFinding(BaseModel):
    photo_summary: str
    claimed_amount: Optional[float]
    typical_range_pkr: Optional[tuple[float, float]]
    cost_ratio: Optional[float]  # claimed / typical maximum
    cost_status: Literal["within", "above", "far_above", "unknown"]
    description_consistent: Optional[bool]  # None if the LLM check was skipped
    description_note: str
    flags: list[Flag]


# ---------------------------------------------------------------------------
# Fraud
# ---------------------------------------------------------------------------
class FraudFinding(BaseModel):
    risk: Literal["low", "medium", "high"]
    signals: list[Flag]


class SpecialistReport(BaseModel):
    """Everything the three agents found for one claim, plus a simple rule-based
    preliminary decision (Phase 4 replaces this with a LangGraph decision agent)."""
    claim_id: str
    coverage: CoverageFinding
    evidence: EvidenceFinding
    fraud: FraudFinding
    preliminary_decision: Literal["approve", "deny", "investigate"]
    decision_reasons: list[str]
