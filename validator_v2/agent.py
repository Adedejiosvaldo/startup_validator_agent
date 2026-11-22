"""Lean startup validator (v2).

Five steps that always run, in order, then a report:

    intake -> market research -> Hormozi -> Africa lens -> verdict -> report

The output is KILL (with the one reason) or TEST (with a one-week experiment
whose only goal is a first payment). Everything else is supporting evidence.

Run from the repo root with `adk web` and pick `validator_v2`.
"""

import os
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools import google_search
from google.genai import types as genai_types

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


# --- Step 5: Verdict ----------------------------------------------------------
verdict_agent = LlmAgent(
    name="verdict",
    model=MODEL,
    description="Decides KILL or TEST and designs the one-week money test.",
    instruction=STANCE
    + """
Decide KILL or TEST.

Founder context: """
    + FOUNDER_CONTEXT
    + """

The idea brief:
{idea_brief}

Market research:
{market_research}

Hormozi assessment:
{hormozi}

Africa lens:
{africa_lens}

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

Don't hedge. If it's close, pick TEST and make the pass threshold strict.
""",
    output_schema=Verdict,
    include_contents="none",
    output_key="verdict",
)


# --- Report (no LLM) ----------------------------------------------------------
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
        intake_agent,
        research_agent,
        hormozi_agent,
        africa_agent,
        verdict_agent,
        ReportAgent(name="report"),
    ],
)
