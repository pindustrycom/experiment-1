"""Structured-output contract for Claude's news interpretation (validated with Pydantic)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

EVENT_TYPES = (
    "executive_change", "acquisition_merger", "product_launch", "product_recall", "regulatory_change",
    "lawsuit_investigation", "earnings_surprise", "guidance_change", "supply_chain_disruption",
    "competitive_development", "market_share_change", "consumer_demand_change", "restructuring_layoffs",
    "major_partnership", "debt_financing", "industry_shock", "other", "not_relevant",
)
EventType = Literal[
    "executive_change", "acquisition_merger", "product_launch", "product_recall", "regulatory_change",
    "lawsuit_investigation", "earnings_surprise", "guidance_change", "supply_chain_disruption",
    "competitive_development", "market_share_change", "consumer_demand_change", "restructuring_layoffs",
    "major_partnership", "debt_financing", "industry_shock", "other", "not_relevant",
]
Scope = Literal["company_specific", "both_companies", "industry_wide", "unrelated"]
RelDirection = Literal["a_advantaged", "a_disadvantaged", "b_advantaged", "b_disadvantaged", "symmetric", "unclear"]


class EvidenceQuote(BaseModel):
    quote: str = Field(description="Verbatim excerpt copied exactly from the source document (max ~300 characters).")
    supports: str = Field(description="Which factual claim of the interpretation this quote supports.")


class CompanyExposure(BaseModel):
    side: Literal["a", "b"]
    exposure: Literal["direct", "indirect", "none"]
    effect: Literal["positive", "negative", "mixed", "none", "unclear"]
    rationale: str


class EventInterpretation(BaseModel):
    is_relevant: bool = Field(description="False if the document says nothing economically meaningful about company A or B.")
    is_material: bool = Field(description="True only if the documented facts plausibly affect revenue, costs, risk or strategy.")
    event_type: EventType
    affected_sides: list[Literal["a", "b"]]
    scope: Scope
    relative_direction: RelDirection
    severity: float = Field(description="0-1 materiality of the documented effect; see rubric.")
    asymmetry: float = Field(description="0 = affects both companies equally, 1 = affects only one company.")
    expected_persistence_days: int = Field(description="How long the effect is expected to matter, in calendar days.")
    extraction_confidence: float = Field(description="0-1 confidence that this interpretation is correct given the text.")
    is_new_information: bool = Field(description="False if this repeats a development already listed under prior known events.")
    related_prior_event_id: int | None = Field(description="ID of the prior known event this continues or repeats, else null.")
    summary: str = Field(description="One or two sentences, strictly supported by the quoted evidence.")
    company_exposure: list[CompanyExposure]
    evidence: list[EvidenceQuote]
    shared_industry_effects: str | None = Field(description="Effects that would hit both companies alike, if any.")
    uncertainties: list[str]

    @field_validator("severity", "asymmetry", "extraction_confidence")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("must be within [0, 1]")
        return v

    @field_validator("expected_persistence_days")
    @classmethod
    def _days(cls, v: int) -> int:
        if not 0 <= v <= 3650:
            raise ValueError("must be within [0, 3650]")
        return v


def direction_to_ticker_form(direction: str, ta: str, tb: str) -> str:
    return {
        "a_advantaged": f"{ta}_advantaged", "a_disadvantaged": f"{ta}_disadvantaged",
        "b_advantaged": f"{tb}_advantaged", "b_disadvantaged": f"{tb}_disadvantaged",
    }.get(direction, direction)
