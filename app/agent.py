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
        if callback_context._invocation_context.agent
        else "unknown_agent"
    )
    output = callback_context.latest_output

    if not isinstance(output, BaseModel):
        return

    validation_results = callback_context.state.get("validation_results", {})
    validation_results[agent_name] = output.dict()
    callback_context.state["validation_results"] = validation_results


def enhance_validation_callback(callback_context: CallbackContext) -> None:
    """Enhanced callback that tracks validation completeness and quality."""
    # Get agent name from the invocation context
    agent_name = (
        callback_context._invocation_context.agent.name
        if callback_context._invocation_context.agent
        else "unknown_agent"
    )
    output = callback_context.latest_output

    # Track all outputs, structured and unstructured
    all_results = callback_context.state.get("all_validation_outputs", {})

    if isinstance(output, BaseModel):
        all_results[agent_name] = {
            "type": "structured",
            "data": output.dict(),
            "agent_name": agent_name,
        }
        # Also update the validation_results for structured outputs
        validation_results = callback_context.state.get("validation_results", {})
        validation_results[agent_name] = output.dict()
        callback_context.state["validation_results"] = validation_results
    else:
        # Handle text outputs from agents like market_research, idea_critique, etc.
        all_results[agent_name] = {
            "type": "text",
            "data": str(output) if output else "",
            "agent_name": agent_name,
        }

    callback_context.state["all_validation_outputs"] = all_results


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
    include_contents="none",
    description="""
    An agent that synthesizes all validation data into a comprehensive, final report.
    """,
    instruction="""
    Transform the comprehensive validation data into a polished, professional startup validation report.

    ---
    ### INPUT DATA
    * Validation Results: `{validation_results}`
    * All Validation Outputs: `{all_validation_outputs}`
    * Enhanced Analysis: `{enhanced_analysis}`

    ---
    ### REPORT STRUCTURE
    Generate a comprehensive startup validation report with the following sections:

    ## Executive Summary
    - Overall recommendation (Pursue/Refine/Pivot/Abandon)
    - Key strengths and critical weaknesses
    - Confidence level in the recommendation

    ## Validation Scores
    - Market Potential: [score]/10
    - Feasibility: [score]/10
    - Competitive Advantage: [score]/10
    - Founder Fit: [score]/10
    - Scalability: [score]/10
    - **Overall Score: [average]/10**

    ## Investment Perspective
    - VC Verdict: [Invest/Pass]
    - Investment reasoning and key factors

    ## Product-Market Fit Analysis
    - PMF Confidence: [Low/Medium/High]
    - Demand signals and market alignment

    ## Market Intelligence
    - Competitive landscape insights
    - Market opportunities and threats
    - Industry trends and dynamics

    ## Strategic Recommendations
    - Immediate next steps
    - Key assumptions to validate
    - Risk mitigation strategies

    ## Implementation Roadmap
    - MVP recommendations
    - Product development priorities
    - Go-to-market approach

    Format as a professional business report with clear headings, bullet points, and actionable insights.
    """,
    output_key="final_startup_report",
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

# Full validation pipeline with quality control
startup_validator_pipeline = SequentialAgent(
    name="startup_validator_pipeline",
    description="""
    A comprehensive startup validation pipeline that performs iterative analysis
    with quality control and generates a professional validation report.
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
        summary_agent,
    ],
)

# Interactive validation agent (manager pattern)
interactive_startup_validator = LlmAgent(
    name="interactive_startup_validator",
    model="gemini-2.0-flash",
    description="The primary startup validation assistant that guides users through the validation process.",
    instruction="""
    You are a startup validation consultant. Your primary function is to help entrepreneurs
    validate their startup ideas through comprehensive analysis.

    When a user presents a startup idea, you should:
    1. **Acknowledge** the idea and explain the validation process
    2. **Execute** comprehensive validation using the startup_validator_pipeline
    3. **Present** the results in a clear, actionable format

    Always be encouraging while being honest about potential challenges.
    Guide users through next steps based on the validation results.
    """,
    sub_agents=[startup_validator_pipeline],
    tools=[AgentTool(core_validation_pipeline)],
)

root_agent = interactive_startup_validator
