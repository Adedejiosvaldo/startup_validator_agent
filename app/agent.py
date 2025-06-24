import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.planners import BuiltInPlanner
from google.genai import types as genai_types
from google.adk.agents.callback_context import CallbackContext
from pydantic import BaseModel, Field
from typing import Literal

# --- Structured Output Models ---
class ScoringResult(BaseModel):
    """Model for the output of the scoring agent."""
    market_potential: int = Field(description="Score for market potential (1-10)")
    feasibility: int = Field(description="Score for feasibility of execution (1-10)")
    competition: int = Field(description="Score for competitive advantage (1-10)")
    founder_fit: int = Field(description="Score for founder's alignment and strength (1-10)")
    scalability: int = Field(description="Score for scalability (1-10)")
    rationale: str = Field(description="Rationale for the scores.")

class InvestorVerdict(BaseModel):
    """Model for the output of the investor agent."""
    verdict: Literal["invest", "pass"] = Field(description="The investment decision.")
    reasoning: str = Field(description="The reasoning behind the verdict.")

class PmfConfidence(BaseModel):
    """Model for the output of the PMF agent."""
    confidence: Literal["Low", "Medium", "High"] = Field(description="The product-market fit confidence rating.")
    analysis: str = Field(description="The analysis behind the confidence rating.")


# --- Callbacks ---
def collect_validation_results_callback(callback_context: CallbackContext) -> None:
    """Collects the structured outputs from the validation agents."""
    agent_name = callback_context.agent.name
    output = callback_context.latest_output
    
    if not isinstance(output, BaseModel):
        return

    validation_results = callback_context.state.get("validation_results", {})
    validation_results[agent_name] = output.dict()
    callback_context.state["validation_results"] = validation_results


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
    tools=[google_search],
    output_schema=ScoringResult,
    after_agent_callback=collect_validation_results_callback,
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
    after_agent_callback=collect_validation_results_callback,
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
    after_agent_callback=collect_validation_results_callback,
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
)


startup_validator_pipeline = SequentialAgent(
    name="startup_validator_pipeline",
    description="""
    A pipeline that validates startup ideas through various lenses, including
    customer pain points, product-market fit, and investment potential.
    """,
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


# Summary Agent
summary_agent = LlmAgent(
    name="startup_summary_agent",
    model="gemini-2.5-flash",
    description="""
    An agent that takes the outputs from all validation agents and synthesizes
    them into a comprehensive, structured report for easy decision-making.
    """,
    instruction="""
    You are a business analyst. Consolidate all the feedback from the validation pipeline into a single, well-structured report that includes:

    ## Executive Summary
    - Overall recommendation (Pursue/Refine/Pivot/Abandon)
    - Key strengths and critical weaknesses

    ## Market Analysis
    - Market size and opportunity
    - Competitive landscape summary
    - Key trends and insights

    ## Product & Strategy
    - Core value proposition
    - MVP recommendations
    - Product-market fit assessment

    ## Investment Perspective
    - Funding potential and investor appeal
    - Key risks and mitigation strategies

    ## Scoring Summary
    - Overall score and breakdown by category
    - Critical success factors

    ## Next Steps
    - Immediate action items
    - Key assumptions to validate

    Format the output as a professional business report with clear sections and actionable insights.
    """,
    sub_agents=[startup_validator_pipeline],
)


root_agent = summary_agent
