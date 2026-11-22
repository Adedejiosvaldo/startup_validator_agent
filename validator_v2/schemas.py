"""Structured outputs for each step of the lean validator.

Every LLM step writes one of these into session state (via output_key), so the
next step and the final report read structured data, not free text.
"""

from typing import Literal

from pydantic import BaseModel, Field


class IdeaBrief(BaseModel):
    """Step 1: the idea, pinned down into the parts that decide whether it sells."""

    one_liner: str = Field(description="The idea in one plain sentence.")
    buyer: str = Field(
        description="The specific person or business who would PAY. Not 'users' or 'SMEs' — "
        "a reachable group, e.g. 'Lagos fintech compliance officers'."
    )
    problem: str = Field(description="The painful problem the buyer has today.")
    current_alternative: str = Field(
        description="What the buyer does about it today (including doing nothing, Excel, "
        "a consultant, a cousin)."
    )
    offer: str = Field(description="What exactly the buyer gets.")
    proposed_price: str = Field(
        description="Price the founder stated, or 'not stated'."
    )
    market: str = Field(description="Country/region the idea starts in.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Anything you had to guess because the founder didn't say it.",
    )


class ValueEquation(BaseModel):
    """Hormozi's Value Equation, scored 1-10 per term."""

    dream_outcome: int = Field(
        ge=1, le=10, description="How badly the buyer wants the result (10 = desperate)."
    )
    perceived_likelihood: int = Field(
        ge=1,
        le=10,
        description="How much the buyer will believe it works for them (10 = certain).",
    )
    time_delay: int = Field(
        ge=1,
        le=10,
        description="How long until the buyer sees the result (1 = instant, 10 = months).",
    )
    effort_and_sacrifice: int = Field(
        ge=1,
        le=10,
        description="How much work/pain the buyer must put in (1 = none, 10 = huge).",
    )
    weakest_term: str = Field(description="Which term drags value down most, and why.")


class HormoziAssessment(BaseModel):
    """Step 3: value, offer and lead channel through the Hormozi books."""

    value_equation: ValueEquation
    grand_slam_offer: str = Field(
        description="A draft offer so good the buyer feels stupid saying no: core promise, "
        "bonuses that remove obstacles, a guarantee, and urgency or scarcity."
    )
    guarantee: str = Field(description="The specific guarantee to attach to the offer.")
    attraction_offer: str = Field(
        description="The first thing to sell to get cash in quickly (Money Models), "
        "before any subscription or upsell."
    )
    best_lead_channel: Literal["warm_outreach", "cold_outreach", "content", "paid_ads"] = (
        Field(description="Which of the Core Four the founder should start with.")
    )
    channel_reason: str = Field(description="Why that channel, for this buyer, right now.")


class AfricaLens(BaseModel):
    """Step 4: the things that kill African startups that a US-centric analysis misses."""

    trust_barrier: str = Field(
        description="Why the buyer might not trust this enough to pay, and what earns that trust."
    )
    distribution: str = Field(
        description="The realistic path to buyers: WhatsApp, agents, associations, "
        "partners, churches, banks, etc."
    )
    payments: str = Field(
        description="How the buyer will actually pay (transfer, Paystack, USSD, cash) "
        "and any friction."
    )
    missing_infrastructure: str = Field(
        description="Infrastructure the idea assumes exists but may not (data, ID, "
        "addresses, credit history, power, internet)."
    )
    regulation: str = Field(
        description="Licences or regulators involved (CBN, NDPC, SEC, NAICOM, etc.), "
        "or 'none significant'."
    )
    hardest_blocker: str = Field(description="The single biggest obstacle from the above.")
    blocker_severity: Literal["none", "manageable", "fatal"] = Field(
        description="'fatal' only if a solo founder cannot realistically get past it."
    )


class TestPlan(BaseModel):
    """A one-week experiment whose only goal is a first payment."""

    buyer: str = Field(description="Exactly who to contact.")
    where_to_find_them: str = Field(description="Exactly where to find 20+ of them this week.")
    offer: str = Field(description="What to offer them, in one or two sentences.")
    price_ngn: int = Field(ge=0, description="Price to ask for, in naira.")
    people_to_contact: int = Field(ge=1, description="How many buyers to pitch.")
    pass_threshold: str = Field(
        description="What counts as a pass, in payments, e.g. '3 of 20 pay'."
    )
    first_message: str = Field(
        description="The exact first message to send, ready to paste into WhatsApp/DM/email."
    )
    days: int = Field(ge=1, le=14, description="Length of the test in days.")


class Verdict(BaseModel):
    """Step 5: the decision."""

    decision: Literal["KILL", "TEST"]
    reason: str = Field(description="One sentence. The real reason.")
    evidence: list[str] = Field(
        description="2-4 points from the research and analysis that back the decision."
    )
    test: TestPlan | None = Field(
        default=None, description="Required when decision is TEST; null when KILL."
    )
    what_would_change_my_mind: str = Field(
        description="For KILL: the evidence that would revive it. For TEST: the result "
        "that should make the founder stop."
    )
