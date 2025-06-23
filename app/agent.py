import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

# MarketResearch Agent
# This agent is responsible for conducting market research, gathering data on competitors, and identifying trends in the industry.
market_research_agent = LlmAgent(
    name="startup_market_research_agent",
    model="gemini-2.5-flash-exp",
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
    name="startup_idea_critique_agent",
    model="gemini-2.5-flash-exp",
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
    model="gemini-2.5-flash-exp",
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
    model="gemini-2.5-flash-exp",
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

# Investor Agent

# Possible Product Market Fit Agent


# customer painpoint
# root_agent = LlmAgent(
#     name="basic_search_agent",
#     model="gemini-2.0-flash-exp",
#     description="A basic search agent that can answer questions using Google Search.",
#     tools=[google_search],
#     instruction="You are an expert researcher. You always stick to the facts and never make up information. If you don't know the answer, you will say 'I don't know'. You will always use Google Search to find the answer to the question.",
# )
