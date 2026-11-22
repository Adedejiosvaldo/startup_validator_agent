"""Lean startup validator (v2).

Runs in order, every time:

    history -> intake -> market research -> Hormozi -> Africa lens
    -> debate (case to kill || case to test) -> verdict x3 -> majority vote -> report

The output is KILL (with the one reason) or TEST (with a one-week experiment
whose only goal is a first payment), plus how many verdict runs agreed.
Every run is logged; recorded test results feed back into future verdicts.

Run from the repo root with `adk web` and pick `validator_v2`.
"""

import logging
import os
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools import google_search
from google.genai import types as genai_types

from .consensus import aggregate_verdicts
from .outcomes import append_record, load_records, summarize_for_prompt
from .report import render_report
from .schemas import AfricaLens, HormoziAssessment, IdeaBrief, Verdict

MODEL = os.getenv("VALIDATOR_MODEL", "gemini-2.5-flash")

# Who is running the test. The verdict step sizes the experiment to this.
FOUNDER_CONTEXT = os.getenv(
    "FOUNDER_CONTEXT",
    "Solo technical founder in Lagos, Nigeria, with a full-time job. "
    "Can spend evenings and weekends, and under N50,000 on any single test.",
)

# Shared stance for every analysis step.
STANCE = """
You are part of a startup validator for an African founder. Be blunt. The founder
wants the truth, not encouragement. Reason from Nigeria/Africa first: assume the
buyer is there unless the brief says otherwise, think in naira, and treat trust,
distribution and payments as first-class problems, not footnotes. Do not judge an
idea by US VC standards. A business that makes N5m a month profitably is a win.
"""


# --- Step 1: Intake -----------------------------------------------------------
intake_agent = LlmAgent(
    name="intake",
    model=MODEL,
    description="Turns a raw idea into a structured brief.",
    instruction=STANCE
    + """
Read the founder's idea and fill in the brief.

- The buyer must be someone who PAYS and who you could find this week. If the founder
  says "users" or "SMEs", narrow it to the most likely paying group.
- If the founder didn't say something, make the most reasonable guess and list it
  under assumptions. Don't ask questions; the founder will correct you after.
""",
    output_schema=IdeaBrief,
    output_key="idea_brief",
)


# --- Step 2: Market research (the only step with web access) -------------------
research_agent = LlmAgent(
    name="market_research",
    model=MODEL,
    description="Finds competitors and evidence that people already pay for this.",
    instruction=STANCE
    + """
The idea brief:
{idea_brief}

Use Google Search. Search for Nigerian and African competitors first, then global.

Report, in short markdown:
1. **Competitors**: up to 5, with what they charge if you can find it. Include the
   current alternative from the brief.
2. **Evidence people pay today**: proof that the buyer already spends money or real
   time on this problem. This matters more than market-size numbers.
3. **Red flags**: dead competitors, failed attempts, or signs nobody wants this.

Name your sources inline. If you can't find something, say so. Never invent figures.
""",
    tools=[google_search],
    include_contents="none",
    output_key="market_research",
)


# --- Step 3: Hormozi ----------------------------------------------------------
hormozi_agent = LlmAgent(
    name="hormozi",
    model=MODEL,
    description="Value Equation, Grand Slam Offer and lead channel from the Hormozi books.",
    instruction=STANCE
    + """
Use Alex Hormozi's books ($100M Offers, $100M Leads, $100M Money Models) as the
source of truth.

The idea brief:
{idea_brief}

Market research:
{market_research}

1. Score the Value Equation from the BUYER's point of view:
   value = (dream outcome x perceived likelihood) / (time delay x effort and sacrifice).
   Be harsh. An idea where the buyer must change habits for months scores high on effort.
2. Draft a Grand Slam Offer for this buyer: the core promise, bonuses that remove each
   obstacle, a guarantee, and real urgency or scarcity.
3. Pick the attraction offer: the first thing to sell for cash quickly.
4. Pick ONE of the Core Four lead channels to start with. A founder with no audience
   and no ad budget usually starts with warm or cold outreach.
""",
    output_schema=HormoziAssessment,
    include_contents="none",
    output_key="hormozi",
)


# --- Step 4: Africa lens ------------------------------------------------------
africa_agent = LlmAgent(
    name="africa_lens",
    model=MODEL,
    description="Trust, distribution, payments, missing infrastructure and regulation.",
    instruction=STANCE
    + """
Look at this idea the way Iyin Aboyeji looks at African startups: which missing
infrastructure layer does it depend on or fill, and how does it win distribution
and trust before anything else?

The idea brief:
{idea_brief}

Market research:
{market_research}

Mark blocker_severity "fatal" only when a solo founder can't realistically get past it,
for example a CBN licence for lending or deposits, or a dependency on data that
doesn't exist.
""",
    output_schema=AfricaLens,
    include_contents="none",
    output_key="africa_lens",
)


# --- Step 5: Debate -----------------------------------------------------------
# Two agents argue opposite sides in parallel, so the verdict has to weigh a real
# case against the idea instead of drifting toward encouragement.
EVIDENCE_IN = """
The idea brief:
{idea_brief}

Market research:
{market_research}

Hormozi assessment:
{hormozi}

Africa lens:
{africa_lens}
"""

bear_agent = LlmAgent(
    name="case_to_kill",
    model=MODEL,
    description="Makes the strongest honest case to kill the idea.",
    instruction=STANCE
    + EVIDENCE_IN
    + """
Make the strongest honest case that the founder should KILL this idea and spend
their evenings on something else. Use only the evidence above; invent nothing.
Up to 5 short bullets, strongest first.
""",
    include_contents="none",
    output_key="bear_case",
)

bull_agent = LlmAgent(
    name="case_to_test",
    model=MODEL,
    description="Makes the strongest honest case to test the idea this week.",
    instruction=STANCE
    + EVIDENCE_IN
    + """
Make the strongest honest case that someone would PAY for this within a week if
the founder pitched it well. "People would like it" doesn't count; only signs of
real spending or urgent pain do. Use only the evidence above; invent nothing.
Up to 5 short bullets, strongest first.
""",
    include_contents="none",
    output_key="bull_case",
)

debate = ParallelAgent(
    name="debate",
    description="Case to kill and case to test, argued in parallel.",
    sub_agents=[bear_agent, bull_agent],
)


# --- Step 6: Verdict, run several times independently -------------------------
VERDICT_RUNS = max(1, int(os.getenv("VERDICT_RUNS", "3")))

VERDICT_INSTRUCTION = (
    STANCE
    + """
Decide KILL or TEST.

Founder context: """
    + FOUNDER_CONTEXT
    + EVIDENCE_IN
    + """
Case to kill:
{bear_case}

Case to test:
{bull_case}

This founder's past test results (real outcomes, not predictions):
{past_outcomes}

Judge between the two cases. The case to kill wins unless the case to test shows
real evidence that the buyer already spends money or effort on this problem.
Use the past results to calibrate: if similar ideas, buyers or prices failed, raise
the bar; if they passed, note what worked and reuse it in the test.

KILL if ANY of these is true:
- No evidence that the buyer spends money or real effort on this problem today.
- The founder can't reach 20 buyers within a week through the chosen channel.
- The Africa lens found a fatal blocker.
- The value is weak on the buyer's side and no offer fixes it.

Otherwise TEST. The test has ONE goal: get someone to pay. Not sign-ups, not
waitlists, not "would you use this?" surveys. Design it so the founder can run it
alone, in their spare time, within budget:
- Sell before building where possible (pre-sale, manual concierge version, deposit).
- Use the Grand Slam Offer and the chosen lead channel.
- Price in naira and ask for real money.
- Write the exact first message in plain, natural language for that channel.
- Make the pass threshold strict.
"""
)


def _make_verdict_agent(i: int) -> LlmAgent:
    # Each run needs its own agent instance (an ADK agent can have only one parent).
    return LlmAgent(
        name=f"verdict_{i}",
        model=MODEL,
        description="Decides KILL or TEST and designs the one-week money test.",
        instruction=VERDICT_INSTRUCTION,
        output_schema=Verdict,
        include_contents="none",
        output_key=f"verdict_{i}",
        # Some randomness, so agreement between runs actually means something.
        generate_content_config=genai_types.GenerateContentConfig(temperature=1.0),
    )


verdict_panel = ParallelAgent(
    name="verdict_panel",
    description="Independent verdict runs, combined by majority vote.",
    sub_agents=[_make_verdict_agent(i) for i in range(1, VERDICT_RUNS + 1)],
)


# --- Deterministic steps (no LLM) ---------------------------------------------
def _state_event(ctx: InvocationContext, author: str, delta: dict) -> Event:
    return Event(
        author=author,
        invocation_id=ctx.invocation_id,
        actions=EventActions(state_delta=delta),
    )


class HistoryAgent(BaseAgent):
    """Loads recorded test results so the verdict can calibrate against them."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        summary = summarize_for_prompt(load_records())
        yield _state_event(ctx, self.name, {"past_outcomes": summary})


class ConsensusAgent(BaseAgent):
    """Majority vote across the verdict runs, then logs the run for later results."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        runs = [state.get(f"verdict_{i}") for i in range(1, VERDICT_RUNS + 1)]
        verdict = aggregate_verdicts(runs)
        delta = {"verdict": verdict}
        if verdict:
            try:
                delta["run_id"] = append_record(state.get("idea_brief") or {}, verdict)
            except OSError as e:  # logging must never cost the user their verdict
                logging.warning("Could not log validation run: %s", e)
        yield _state_event(ctx, self.name, delta)


class ReportAgent(BaseAgent):
    """Renders the final markdown from state, so the report can't add new claims."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        report = render_report(dict(ctx.session.state))
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=report)]
            ),
        )


root_agent = SequentialAgent(
    name="lean_startup_validator",
    description="Validates a startup idea and returns KILL or a one-week money test.",
    sub_agents=[
        HistoryAgent(name="history"),
        intake_agent,
        research_agent,
        hormozi_agent,
        africa_agent,
        debate,
        verdict_panel,
        ConsensusAgent(name="consensus"),
        ReportAgent(name="report"),
    ],
)
