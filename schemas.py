"""
Pydantic schemas: the exact shape of data the intake agent must return.

Tip: every field is required but may be null (Optional without a default).
That keeps the schemas compatible with OpenAI's strict structured output,
and forces the model to say "null" instead of silently skipping a field.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

DamageType = Literal["dent", "scratch", "crack", "glass shatter", "lamp broken",
                     "tire flat", "none", "other"]
Severity = Literal["minor", "moderate", "severe", "none"]


# ---------------------------------------------------------------------------
# What the LLM extracts
# ---------------------------------------------------------------------------
class ClaimForm(BaseModel):
    """Data extracted from claim_form.pdf."""
    claim_number: Optional[str] = Field(description="Claim number, e.g. C002")
    date_submitted: Optional[str] = Field(description="Date submitted, YYYY-MM-DD")
    claimant_name: Optional[str] = Field(description="Full name of the claimant")
    cnic: Optional[str] = Field(description="CNIC number exactly as written")
    phone: Optional[str] = Field(description="Phone number exactly as written")
    policy_number: Optional[str] = Field(description="Insurance policy number")
    vehicle: Optional[str] = Field(description="Vehicle make and model")
    registration_number: Optional[str] = Field(description="Vehicle registration number")
    incident_date: Optional[str] = Field(description="Date of incident, YYYY-MM-DD")
    location: Optional[str] = Field(description="Where the incident happened")
    description: Optional[str] = Field(description="Claimant's description of the incident, verbatim")
    amount_claimed: Optional[float] = Field(description="Amount claimed as a number in PKR, no commas or currency")


class EstimateLine(BaseModel):
    item: str = Field(description="Line item description")
    amount: float = Field(description="Amount in PKR as a number")


class RepairEstimate(BaseModel):
    """Data extracted from repair_estimate.pdf."""
    workshop: Optional[str] = Field(description="Workshop / garage name")
    invoice_number: Optional[str] = Field(description="Invoice or estimate number")
    estimate_date: Optional[str] = Field(description="Estimate date, YYYY-MM-DD")
    customer_name: Optional[str] = Field(description="Customer name on the estimate")
    vehicle: Optional[str] = Field(description="Vehicle make and model on the estimate")
    registration_number: Optional[str] = Field(description="Registration number on the estimate")
    line_items: list[EstimateLine] = Field(description="Every line item except the total")
    total: Optional[float] = Field(description="TOTAL amount as a number in PKR")


class DamageItem(BaseModel):
    """One separate damage visible in the photo."""
    part: str = Field(description="Car part, e.g. 'rear bumper', 'front left door', 'headlight'")
    damage_type: DamageType = Field(description="Type of this damage")
    severity: Severity = Field(description="Severity of this damage")


class PhotoAssessment(BaseModel):
    """What the vision model sees in the damage photo."""
    is_vehicle_photo: bool = Field(description="True if the photo shows a car or part of a car")
    all_damages: list[DamageItem] = Field(
        description="EVERY separate damage visible, one item each (e.g. a dent AND the scratches on it "
                    "are two items). Empty if no damage.")
    damaged_part: Optional[str] = Field(description="Part with the main damage (chosen by the priority rules)")
    damage_type: DamageType = Field(description="Main damage type, chosen by the priority rules")
    severity: Severity = Field(description="Severity of the main damage")
    description: str = Field(description="One or two sentences describing only what is visible")
    image_quality_ok: bool = Field(description="False if the photo is too blurry, dark or cropped to judge")


# ---------------------------------------------------------------------------
# What the intake step returns
# ---------------------------------------------------------------------------
class ValidationIssue(BaseModel):
    code: str           # short machine-readable code, e.g. "amount_mismatch"
    severity: Literal["info", "warning", "error"]
    message: str        # human-readable explanation


class IntakeResult(BaseModel):
    claim_id: str
    claim_form: ClaimForm
    estimate: RepairEstimate
    photo: PhotoAssessment
    issues: list[ValidationIssue]
    missing_fields: list[str]
    ready_for_review: bool  # False if there are errors or missing required fields
