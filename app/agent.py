import logging
from collections.abc import AsyncGenerator
from typing import Literal

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from pydantic import BaseModel, Field


# --- Structured Output Models ---
class ScoringResult(BaseModel):
    """Model for the output of the scoring agent."""

    market_potential: int = Field(description="Score for market potential (1-10)")
    feasibility: int = Field(description="Score for feasibility of execution (1-10)")
    competition: int = Field(description="Score for competitive advantage (1-10)")
    founder_fit: int = Field(
        description="Score for founder's alignment and strength (1-10)"
    )
    scalability: int = Field(description="Score for scalability (1-10)")
    rationale: str = Field(description="Rationale for the scores.")


class InvestorVerdict(BaseModel):
    """Model for the output of the investor agent."""

    verdict: Literal["invest", "pass"] = Field(description="The investment decision.")
    reasoning: str = Field(description="The reasoning behind the verdict.")


class PmfConfidence(BaseModel):
    """Model for the output of the PMF agent."""

    confidence: Literal["Low", "Medium", "High"] = Field(
        description="The product-market fit confidence rating."
    )
    analysis: str = Field(description="The analysis behind the confidence rating.")


class ValidationFeedback(BaseModel):
    """Model for providing evaluation feedback on validation quality."""

    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if validation is sufficient, 'fail' if needs improvement."
    )
    comment: str = Field(
        description="Detailed explanation of the evaluation, highlighting strengths and/or weaknesses."
    )
    improvement_areas: list[str] | None = Field(
        default=None,
        description="List of specific areas that need improvement. Should be null or empty if grade is 'pass'.",
    )


# --- Callbacks ---
def collect_validation_results_callback(callback_context: CallbackContext) -> None:
    """Collects the structured outputs from the validation agents."""
    # Get agent name from the invocation context
    agent_name = (
        callback_context._invocation_context.agent.name
        if callback_context._invocation_context.agent.name
        else "unknown_agent"
    )

    # Get the latest event from the session
    session = callback_context._invocation_context.session
    if not session.events:
        return

    latest_event = session.events[-1]
    if not latest_event.content or not latest_event.content.parts:
        return

    # Try to parse the output as a Pydantic model
    try:
        output_text = latest_event.content.parts[0].text
        # For now, we'll store the text output - structured parsing would need the specific model
        validation_results = callback_context.state.get("validation_results", {})
        validation_results[agent_name] = {"output": output_text}
        callback_context.state["validation_results"] = validation_results
    except (AttributeError, IndexError):
        return


def enhance_validation_callback(callback_context: CallbackContext) -> None:
    """Enhanced callback that tracks validation completeness and quality."""
    # Get agent name from the invocation context
    agent_name = (
        callback_context._invocation_context.agent.name
        if callback_context._invocation_context.agent
        else "unknown_agent"
    )

    # Get the latest event from the session
    session = callback_context._invocation_context.session
    if not session.events:
        return

    latest_event = session.events[-1]
    if not latest_event.content or not latest_event.content.parts:
        return

    try:
        output_text = latest_event.content.parts[0].text

        # Track all outputs, structured and unstructured
        all_results = callback_context.state.get("all_validation_outputs", {})

        all_results[agent_name] = {
            "type": "text",
            "data": output_text,
            "agent_name": agent_name,
        }

        callback_context.state["all_validation_outputs"] = all_results

        # Also update validation_results for consistency
        validation_results = callback_context.state.get("validation_results", {})
        validation_results[agent_name] = {"output": output_text}
        callback_context.state["validation_results"] = validation_results

        # Debug logging
        print(
            f"[CALLBACK DEBUG] Captured output from {agent_name}: {len(output_text)} characters"
        )

    except (AttributeError, IndexError) as e:
        print(f"[CALLBACK ERROR] Failed to extract output from {agent_name}: {e}")
        return


# --- Custom Agent for Loop Control ---
class ValidationQualityChecker(BaseAgent):
    """Checks validation quality and escalates to stop the loop if grade is 'pass'."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get("validation_evaluation")
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(
                f"[{self.name}] Validation evaluation passed. Escalating to stop loop."
            )
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(
                f"[{self.name}] Validation evaluation failed or not found. Loop will continue."
            )
            yield Event(author=self.name)


# MarketResearch Agent
# This agent is responsible for conducting market research, gathering data on competitors, and identifying trends in the industry.
market_research_agent = LlmAgent(
    name="startup_market_research_agent",
    model="gemini-2.5-flash",
    description="""
    A specialized agent for conducting in-depth market research to support startup idea validation.
    It gathers and analyzes data on competitors, industry trends, venture capital interest, and
    emerging market dynamics. This agent plays a critical role in helping founders understand
    market viability, gaps, and opportunities before pursuing a startup idea.
    """,
    instruction="""
    You are an expert market researcher and startup analyst. Your primary objective is to evaluate
    the potential of a startup idea by performing deep market research. You must:

    1. Extract key industry-related keywords from the provided idea or concept.
    2. Use the Google Search tool to gather real-time data on:
       - Direct and indirect competitors
       - Industry trends and patterns
       - Recent venture capital funding and investor interest
       - Market size, growth rate, and potential gaps
    3. Summarize findings in a structured and actionable format, highlighting threats, opportunities, and whitespace.
    4. Focus on providing insights that would help validate or refine the startup idea.

    Always rely on Google Search for the most up-to-date market information, and make sure to extract keywords that best represent the core of the idea.
    """,
    tools=[google_search],
    after_agent_callback=enhance_validation_callback,
)


# Idea Critique Agent
idea_critique_agent = LlmAgent(
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    name="startup_idea_critique_agent",
    model="gemini-2.5-flash",
    description="""
    An agent that critically evaluates startup ideas based on innovation,
    feasibility, market relevance, and differentiation. Provides constructive
    feedback and practical suggestions to refine the concept.
    """,
    instruction="""
    You are a seasoned startup mentor. Your role is to evaluate startup ideas and provide:
    - Constructive feedback
    - Suggestions to improve the business model
    - Warnings about unrealistic assumptions
    - Insights into potential competitive or execution risks
    Analyze based on viability, novelty, and scalability.
    """,
    after_agent_callback=enhance_validation_callback,
)


# Product Development Agent
product_dev_agent = LlmAgent(
    name="startup_product_development_agent",
    model="gemini-2.5-flash",
    description="""
    A product development agent that transforms validated startup ideas
    into clear product visions, roadmaps, and feature sets.
    """,
    instruction="""
    You are a product strategist. Based on the startup idea:
    - Define the core value proposition
    - Propose key features and differentiators
    - Map out a phased development roadmap
    - Recommend tech stacks or approaches where needed
    """,
    after_agent_callback=enhance_validation_callback,
)

# MVP Agent
mvp_agent = LlmAgent(
    name="startup_mvp_agent",
    model="gemini-2.5-flash",
    description="""
    An MVP planning agent that helps founders design the simplest version
    of their product that can test key assumptions and deliver core value.
    """,
    instruction="""
    Your task is to break down the product vision into a minimal viable product (MVP).
    - Identify core features to test the main hypothesis
    - Recommend quick-to-build versions of each component
    - Ensure the MVP is aligned with the target audience and value proposition
    """,
    tools=[google_search],
    after_agent_callback=enhance_validation_callback,
)

# Scoring Agent
scoring_agent = LlmAgent(
    name="startup_scoring_agent",
    model="gemini-2.5-flash",
    description="""
    A scoring agent that evaluates startup ideas based on predefined metrics such
    as market size, competition, feasibility, scalability, and uniqueness.
    """,
    instruction="""
    Score startup ideas from 1 to 10 across these dimensions:
    - Market Potential
    - Feasibility of Execution
    - Competitive Advantage
    - Founder's Alignment/Strength
    - Scalability
    Provide a rationale for each score.
    """,
    # tools=[google_search],
    output_schema=ScoringResult,
    after_agent_callback=enhance_validation_callback,
)


# Investor Agent
investor_agent = LlmAgent(
    name="startup_investor_agent",
    model="gemini-2.5-flash",
    description="""
    An investor simulation agent that evaluates the startup from the lens of a VC,
    giving feedback on investment potential and highlighting red or green flags.
    """,
    instruction="""
    You are a seasoned venture capitalist. Evaluate startup ideas by:
    - Identifying investable signals
    - Pointing out deal-breaking flaws
    - Estimating the TAM and exit potential
    - Giving a mock "invest or pass" verdict with reasons
    """,
    output_schema=InvestorVerdict,
    after_agent_callback=enhance_validation_callback,
)

# Possible Product Market Fit Agent
pmf_agent = LlmAgent(
    name="startup_pmf_agent",
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    description="""
    An agent that analyzes whether a proposed startup idea is likely to achieve
    product-market fit based on user needs, demand signals, and solution fit.
    """,
    instruction="""
    Determine if there is strong alignment between the target user’s pain points and the proposed solution.
    - Identify demand signals
    - Check how differentiated the solution is
    - Assess willingness to pay or adopt
    Provide a product-market fit confidence rating (Low, Medium, High)
    """,
    output_schema=PmfConfidence,
    after_agent_callback=enhance_validation_callback,
)


# customer painpoint
painpoint_agent = LlmAgent(
    name="startup_customer_painpoint_agent",
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    description="""
    A customer-centric agent that identifies core pain points experienced by the
    target audience relevant to the startup idea.
    """,
    instruction="""
    Extract and analyze customer pain points from the idea and market context.
    Use empathy-driven reasoning to:
    - Identify underserved needs
    - Pinpoint frustrations or inefficiencies
    - Suggest ways the product could directly address these issues
    """,
    after_agent_callback=enhance_validation_callback,
)


# Validation Quality Evaluator Agent
validation_evaluator = LlmAgent(
    model="gemini-2.0-flash",
    name="validation_evaluator",
    description="Critically evaluates the quality and completeness of startup validation.",
    instruction="""
    You are a meticulous startup validation analyst. Your task is to evaluate the quality
    and completeness of the validation analysis contained in 'all_validation_outputs'.

    **CRITICAL RULES:**
    1. Assess the depth and quality of each validation component
    2. Check for consistency across different analyses
    3. Identify gaps in the validation coverage
    4. Ensure all key startup validation areas are adequately addressed

    Focus on evaluating:
    - Completeness: Are all critical aspects covered?
    - Depth: Is the analysis thorough enough for decision-making?
    - Consistency: Do the analyses align with each other?
    - Quality: Are insights actionable and well-reasoned?

    If you find significant gaps or quality issues, assign a grade of "fail" and specify
    improvement areas. If the validation is comprehensive and high-quality, grade "pass".

    Your response must be a single, raw JSON object validating against the 'ValidationFeedback' schema.
    """,
    output_schema=ValidationFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="validation_evaluation",
)

# Enhanced Analysis Agent
enhanced_analysis_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="enhanced_analysis_agent",
    description="Performs additional analysis to address validation gaps.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialist startup analyst executing an enhancement pass.
    You have been activated because the validation was graded as 'fail'.

    1. Review the 'validation_evaluation' state key to understand the feedback and required improvements.
    2. Review the existing 'all_validation_outputs' to understand what has been done.
    3. Address EVERY improvement area listed in the evaluation feedback.
    4. Provide additional analysis, insights, and recommendations to fill the gaps.
    5. Use the Google Search tool if you need current market data or trends.

    Your output should directly address the identified weaknesses and provide the missing analysis.
    """,
    tools=[google_search],
    output_key="enhanced_analysis",
    after_agent_callback=enhance_validation_callback,
)

# Summary Agent
summary_agent = LlmAgent(
    name="startup_summary_agent",
    model="gemini-2.0-flash",
    description="""
    A strategic business analyst that creates compelling, insightful startup validation reports
    by synthesizing diverse validation perspectives into actionable intelligence.
    """,
    instruction="""
    You are an elite startup strategist and business intelligence analyst. Your mission is to transform
    raw validation data into a compelling strategic narrative that guides entrepreneurial decision-making.

    **ACCESS VALIDATION DATA FROM SESSION STATE:**
    - `all_validation_outputs` - comprehensive analysis from all validation agents
    - `validation_evaluation` - quality assessment feedback
    - `enhanced_analysis` - additional insights if improvements were made

    **YOUR STRATEGIC APPROACH:**

    1. **SYNTHESIZE, DON'T SUMMARIZE**: Connect insights across different validation dimensions to reveal hidden patterns, conflicts, and synergies

    2. **TELL A COMPELLING STORY**: Craft a narrative that shows the startup's journey from idea to market reality

    3. **PROVIDE STRATEGIC INTELLIGENCE**: Go beyond data extraction to offer strategic recommendations based on interconnected insights

    4. **IDENTIFY CRITICAL SUCCESS FACTORS**: Pinpoint the 2-3 make-or-break factors that will determine success

    5. **CREATE ACTIONABLE PATHWAYS**: Design concrete next steps that address multiple validation concerns simultaneously

    **CREATIVE SYNTHESIS TECHNIQUES:**
    - Look for convergent themes across different analyses
    - Identify contradictions that reveal hidden risks or opportunities
    - Connect customer pain points directly to competitive advantages
    - Link market trends to implementation strategies
    - Correlate investment appeal with technical feasibility

    **GENERATE A STRATEGIC VALIDATION REPORT:**

    ## 🎯 STRATEGIC VALIDATION NARRATIVE

    **The Big Picture**: Create a compelling 2-3 paragraph story that weaves together market opportunity, customer pain, competitive dynamics, and solution potential. Focus on the strategic narrative rather than listing facts.

    **Critical Success Equation**: Identify the 2-3 interconnected factors that will make or break this venture (e.g., "Success = Enterprise Trust × Technical Differentiation × Go-to-Market Excellence").

    **Strategic Positioning**: Based on all analyses, define where this startup should position itself in the market ecosystem to maximize competitive advantage.

    ## 🔥 CONVERGED INSIGHTS & CONTRADICTIONS

    **Where All Analyses Align**: Highlight the strongest convergent themes across validation dimensions (what everyone agrees on).

    **Critical Tensions**: Identify contradictions between different analyses (e.g., high market potential vs. intense competition) and resolve them with strategic recommendations.

    **Hidden Connections**: Reveal non-obvious links between customer pain points, market gaps, and technical capabilities.

    ## 📊 INTELLIGENT METRICS SYNTHESIS

    **Validation Confidence Score**: [X]/10 - A synthesized score based on convergent evidence across all dimensions

    **Market-Product-Execution Triangle**:
    - Market Readiness: [score] - How ready is the market for this solution?
    - Product Viability: [score] - How feasible is building a differentiated product?
    - Execution Probability: [score] - How likely is successful execution given constraints?

    **Investment Thesis**: Distill the investor perspective into a clear 1-2 sentence thesis

    ## 🚀 STRATEGIC PATHWAY FORWARD

    **The Critical Path**: Design a strategic sequence that addresses multiple validation concerns simultaneously (not just a task list).

    **Validation Cascade**: Identify which assumptions, if proven, would de-risk multiple other concerns.

    **Strategic Experiments**: Propose 2-3 focused experiments that could validate core hypotheses and build competitive advantage.

    ## ⚡ COMPETITIVE BATTLE PLAN

    **Differentiation Strategy**: Based on market analysis and pain points, define how to win against both direct competitors and "build vs. buy" mentality.

    **Moat Building**: Connect MVP features to long-term defensibility.

    **Market Entry Wedge**: Identify the specific market segment/use case that offers the highest probability of initial success.

    ## 🎲 RISK MATRIX & MITIGATION

    **Interconnected Risk Assessment**: Show how different risks compound or mitigate each other.

    **Strategic Risk Mitigation**: Propose strategies that address multiple risk categories simultaneously.

    ## 💡 CREATIVE STRATEGIC OPTIONS

    Based on the complete analysis, propose 2-3 creative strategic alternatives beyond the obvious path (e.g., vertical specialization, partnership models, platform strategies).

    **FORMATTING PRINCIPLES:**
    - Lead with insights, not data
    - Connect dots across different validation dimensions
    - Use strategic frameworks and business concepts
    - Focus on decision-making guidance
    - Make every section actionable and interconnected
    """,
    output_key="final_startup_report",
)

# Final Report Consolidation Agent
final_report_agent = LlmAgent(
    name="final_report_consolidation_agent",
    model="gemini-2.0-flash",
    description="""
    An executive presentation specialist that transforms comprehensive validation analysis
    into compelling, decision-oriented startup intelligence reports.
    """,
    instruction="""
    You are an elite business intelligence specialist who creates executive-level startup reports
    that drive strategic decision-making. Your reports are known for their clarity, insight, and actionability.

    **ACCESS COMPREHENSIVE ANALYSIS:**
    - `all_validation_outputs` - complete validation analysis
    - `strategic_synthesis` - strategic synthesis and breakthrough insights
    - `final_startup_report` - comprehensive validation report

    **YOUR MISSION**: Transform detailed analysis into a compelling executive brief that tells a complete story and guides decision-making.

    **CREATIVE REPORT FORMAT:**

    # 🚀 Startup Intelligence Brief: [STARTUP NAME]

    ## 📋 EXECUTIVE DECISION FRAMEWORK

    **The Bottom Line**: [One compelling sentence: Should you pursue this opportunity?]

    **Strategic Verdict**: PURSUE / REFINE / PIVOT / ABANDON

    **Confidence Level**: [High/Medium/Low] based on convergent evidence

    **Critical Success Factors**: The 2-3 make-or-break elements that determine success

    ## 🎯 THE STRATEGIC STORY

    **Market Moment**: Why now? What market forces create this opportunity?

    **Customer Reality**: What's really driving customer pain and willingness to pay?

    **Competitive Landscape**: Where do you fit and how do you win?

    **Your Advantage**: What makes this venture uniquely positioned to succeed?

    ## 📊 INTELLIGENCE SCORECARD

    | Dimension | Score | Strategic Implication |
    |-----------|-------|---------------------|
    | Market Opportunity | X/10 | [Key insight] |
    | Technical Feasibility | X/10 | [Key insight] |
    | Competitive Position | X/10 | [Key insight] |
    | Execution Probability | X/10 | [Key insight] |
    | **Overall Validation** | **X/10** | **[Overall assessment]** |

    ## 🎪 THE INVESTMENT STORY

    **Investor Perspective**: [Investment verdict and reasoning]

    **Value Creation Thesis**: How this startup creates and captures value

    **Risk-Return Profile**: What investors see as the upside and downside

    ## 🚀 GO-TO-MARKET BLUEPRINT

    **Market Entry Strategy**: How to get your first customers

    **MVP Battle Plan**: What to build first and why

    **Scaling Pathway**: How early success leads to market dominance

    ## ⚡ BREAKTHROUGH OPPORTUNITIES

    **Strategic Insights**: Non-obvious opportunities discovered through synthesis

    **Competitive Jujitsu**: How to turn challenges into advantages

    **Expansion Vectors**: Where this leads in 3-5 years

    ## 🎯 90-DAY ACTION PLAN

    **Week 1-2: Foundation**
    - [ ] [Specific immediate actions]

    **Week 3-8: Validation**
    - [ ] [Key experiments and validations]

    **Week 9-12: Launch Prep**
    - [ ] [Building and preparing for market]

    ## 🚨 CRITICAL WATCH-OUTS

    **Execution Risks**: What could derail progress

    **Market Risks**: External factors to monitor

    **Competitive Threats**: How competitors might respond

    ## 🔮 STRATEGIC OPTIONS

    **Plan A**: [Primary recommended strategy]

    **Plan B**: [Alternative approach if Plan A faces challenges]

    **Plan C**: [Pivot strategy if market conditions change]

    ---

    **Final Word**: [Compelling closing statement that synthesizes everything into clear guidance]

    **DESIGN PRINCIPLES:**
    - Lead with strategic clarity, not data dumps
    - Every section should drive decision-making
    - Use visual elements (emojis, tables, bullets) for impact
    - Connect insights across sections
    - End with clear, actionable guidance
    """,
    output_key="executive_startup_brief",
)

# Strategic Synthesis Agent
strategic_synthesis_agent = LlmAgent(
    name="strategic_synthesis_agent",
    model="gemini-2.0-flash",
    description="""
    A strategic consultant that creates innovative synthesis by connecting disparate validation insights
    into breakthrough strategic recommendations and identifying non-obvious opportunities.
    """,
    instruction="""
    You are a world-class strategy consultant specializing in startup validation and market entry strategies.
    Your expertise lies in seeing patterns others miss and connecting seemingly unrelated insights into breakthrough strategies.

    **ACCESS SESSION STATE FOR SYNTHESIS:**
    - `all_validation_outputs` - comprehensive validation analysis from all agents
    - Review market research, customer pain points, competitive analysis, technical feasibility, and investor perspectives

    **YOUR SYNTHESIS METHODOLOGY:**

    1. **PATTERN RECOGNITION**: Identify recurring themes, contradictions, and hidden connections across all validation dimensions

    2. **STRATEGIC TRIANGULATION**: Where market needs, technical capabilities, and competitive gaps intersect, find the sweet spot

    3. **INNOVATION OPPORTUNITIES**: Look for creative ways to address multiple customer pain points with single solutions

    4. **COMPETITIVE JUJITSU**: Find ways to turn competitive threats into strategic advantages

    5. **MARKET TIMING ANALYSIS**: Assess whether this is the right solution at the right time for the right market

    **GENERATE STRATEGIC SYNTHESIS:**

    ## 🧠 CROSS-DIMENSIONAL INSIGHT SYNTHESIS

    **Convergence Analysis**: What insights appear consistently across different validation perspectives? These represent your strongest foundations.

    **Divergence Insights**: Where do different analyses contradict each other? These often reveal the most critical strategic choices.

    **Emergent Opportunities**: What new possibilities emerge when you combine insights from multiple domains (e.g., customer pain + market trends + technical capabilities)?

    ## 🎯 STRATEGIC POSITIONING MATRIX

    **Competitive Differentiation Formula**: Based on the complete analysis, what's the unique formula for winning? (e.g., "Enterprise Trust × Technical Depth × Industry Expertise")

    **Blue Ocean Potential**: Are there opportunities to create new market categories by combining insights in novel ways?

    **Strategic Moats**: How can early moves create defensible advantages that compound over time?

    ## ⚡ BREAKTHROUGH STRATEGIES

    **The Unconventional Path**: Based on all validation data, what non-obvious strategic approach could leapfrog competitors?

    **Ecosystem Play**: How could this startup position itself as a platform or critical infrastructure rather than just another vendor?

    **Category Creation**: Is there potential to define a new market category that the startup could own?

    ## 🚀 EXECUTION MULTIPLIERS

    **High-Leverage Experiments**: What single experiments could validate multiple assumptions simultaneously?

    **Compound Growth Strategies**: How can initial customer success create exponential growth through network effects, data advantages, or ecosystem lock-in?

    **Strategic Partnerships**: What alliance strategies could accelerate market entry and reduce competition?

    ## 🔮 FUTURE-PROOFING ANALYSIS

    **Trend Convergence**: How do current market trends suggest this opportunity will evolve over 3-5 years?

    **Disruption Resistance**: How defensible is this opportunity against both traditional competitors and emerging technologies?

    **Expansion Vectors**: What adjacent opportunities could this startup capture as it grows?

    **SYNTHESIS PRINCIPLES:**
    - Connect dots across disciplines
    - Challenge conventional wisdom
    - Identify compound advantages
    - Design for optionality
    - Think in systems, not features
    """,
    output_key="strategic_synthesis",
    after_agent_callback=enhance_validation_callback,
)

# Core validation pipeline with iterative improvement
core_validation_pipeline = SequentialAgent(
    name="core_validation_pipeline",
    description="Executes core startup validation across multiple dimensions.",
    sub_agents=[
        idea_critique_agent,
        market_research_agent,
        painpoint_agent,
        pmf_agent,
        investor_agent,
        mvp_agent,
        product_dev_agent,
        scoring_agent,
    ],
)

# Full validation pipeline with quality control, strategic synthesis, and executive reporting
startup_validator_pipeline = SequentialAgent(
    name="startup_validator_pipeline",
    description="""
    A comprehensive startup validation pipeline that performs iterative analysis with quality control,
    strategic synthesis, and generates both detailed and executive-level validation reports.
    """,
    sub_agents=[
        core_validation_pipeline,
        LoopAgent(
            name="validation_quality_loop",
            max_iterations=2,
            sub_agents=[
                validation_evaluator,
                ValidationQualityChecker(name="quality_checker"),
                enhanced_analysis_agent,
            ],
        ),
        strategic_synthesis_agent,
        summary_agent,
        final_report_agent,
    ],
)

# Interactive validation agent (manager pattern)
interactive_startup_validator = LlmAgent(
    name="interactive_startup_validator",
    model="gemini-2.0-flash",
    description="The primary startup validation assistant that guides users through the validation process.",
    instruction="""
    You are a world-class startup validation consultant who guides entrepreneurs through comprehensive
    validation using advanced multi-agent analysis and strategic intelligence synthesis.

    When a user presents a startup idea, you should:

    1. **Acknowledge & Set Expectations**: Welcome the idea and explain you'll conduct a comprehensive
       validation using multiple specialist agents for market research, customer analysis, competitive
       intelligence, technical feasibility, investment perspective, and strategic synthesis.

    2. **Execute Comprehensive Validation**: Run the startup_validator_pipeline which includes:
       - Core validation across 8+ dimensions
       - Quality control with iterative improvement
       - Strategic synthesis for breakthrough insights
       - Executive-level reporting

    3. **Present Strategic Intelligence**: After validation, present both the detailed analysis and
       the executive brief, highlighting:
       - Key strategic insights and breakthrough opportunities
       - Critical success factors and potential pitfalls
       - Actionable next steps with clear prioritization
       - Creative strategic alternatives and positioning options

    **YOUR COMMUNICATION STYLE:**
    - Be encouraging while brutally honest about challenges
    - Focus on strategic insights rather than data dumps
    - Connect validation findings to actionable business decisions
    - Highlight both conventional wisdom and contrarian insights
    - Guide users toward the highest-probability success paths

    Always conclude by asking if they'd like deeper analysis on any specific aspect or help
    developing the recommended strategic pathway.
    """,
    sub_agents=[startup_validator_pipeline],
    tools=[AgentTool(core_validation_pipeline)],
)

root_agent = interactive_startup_validator
