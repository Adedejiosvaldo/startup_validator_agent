import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.planners import BuiltInPlanner
from google.genai import types as genai_types

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

root_agent = startup_validator_pipeline
