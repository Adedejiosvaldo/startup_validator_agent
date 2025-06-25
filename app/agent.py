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


# --- Real-Time Data Integration Models ---
class MarketIntelligence(BaseModel):
    """Model for real-time market data and intelligence."""

    funding_data: dict = Field(
        default_factory=dict,
        description="Recent funding rounds and investment trends",
    )
    patent_landscape: dict = Field(
        default_factory=dict, description="Patent analysis and IP landscape"
    )
    social_sentiment: dict = Field(
        default_factory=dict, description="Social media sentiment and discussions"
    )
    competitive_intelligence: dict = Field(
        default_factory=dict, description="Real-time competitive tracking data"
    )
    market_timing_indicators: list[str] = Field(
        default_factory=list, description="Market timing signals and trends"
    )
    data_freshness: str = Field(description="Timestamp of data collection")
    confidence_score: float = Field(description="Confidence in data quality (0-1)")


class VisualizationRequest(BaseModel):
    """Model for requesting specific visualizations."""

    chart_type: Literal[
        "market_map",
        "competitive_landscape",
        "funding_timeline",
        "customer_journey",
        "feature_matrix",
        "scenario_comparison",
        "risk_heatmap",
        "validation_dashboard",
    ] = Field(description="Type of visualization needed")
    data_source: str = Field(description="Source of data for visualization")
    title: str = Field(description="Title for the visualization")
    description: str = Field(description="Description of what the chart shows")
    insights: list[str] = Field(
        default_factory=list,
        description="Key insights highlighted by the visualization",
    )
    interactive_elements: list[str] = Field(
        default_factory=list, description="Interactive features of the visualization"
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Priority for generating this visualization"
    )


class DataSource(BaseModel):
    """Model for tracking data sources and their reliability."""

    source_name: str = Field(description="Name of the data source")
    source_type: Literal[
        "funding_database",
        "patent_office",
        "social_media",
        "market_research",
        "company_filings",
        "news_api",
    ] = Field(description="Type of data source")
    last_updated: str = Field(description="When data was last refreshed")
    reliability_score: float = Field(description="Reliability rating (0-1)")
    coverage_areas: list[str] = Field(description="What aspects this source covers")
    api_status: Literal["active", "limited", "unavailable"] = Field(
        description="Current status of the data source"
    )


class RealTimeInsight(BaseModel):
    """Model for real-time insights and market signals."""

    insight_type: Literal[
        "funding_trend",
        "competitor_move",
        "market_shift",
        "regulatory_change",
        "technology_breakthrough",
        "customer_signal",
    ] = Field(description="Type of real-time insight")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency level of the insight"
    )
    title: str = Field(description="Brief title of the insight")
    description: str = Field(description="Detailed description of the insight")
    impact_assessment: str = Field(description="How this impacts the startup idea")
    recommended_action: str = Field(description="Recommended response or action")
    data_sources: list[str] = Field(description="Sources supporting this insight")
    timestamp: str = Field(description="When this insight was generated")


# --- Wrapper Models for List Outputs ---
class RealTimeInsights(BaseModel):
    """Wrapper model for multiple real-time insights."""

    insights: list[RealTimeInsight] = Field(
        default_factory=list,
        description="List of real-time insights and market signals",
    )


class VisualizationRequests(BaseModel):
    """Wrapper model for multiple visualization requests."""

    requests: list[VisualizationRequest] = Field(
        default_factory=list, description="List of visualization requests"
    )


class DataSources(BaseModel):
    """Wrapper model for multiple data sources."""

    sources: list[DataSource] = Field(
        default_factory=list,
        description="List of data sources and their reliability information",
    )


# --- Enhanced Structured Output Models for UX Improvements ---
class IdeaProfile(BaseModel):
    """Model for capturing detailed idea characteristics and context."""

    idea_description: str = Field(description="The core startup idea description")
    idea_maturity: Literal["concept", "prototype", "early_traction", "scaling"] = Field(
        description="Stage of idea development"
    )
    industry_category: str = Field(description="Primary industry or category")
    target_market: str = Field(description="Target customer segment")
    founder_background: str = Field(description="Founder's experience and expertise")
    specific_concerns: list[str] = Field(
        default_factory=list,
        description="Specific areas the founder is most worried about",
    )
    business_model: str | None = Field(
        default=None, description="Proposed business model"
    )
    competitive_landscape_known: bool = Field(
        default=False, description="Whether founder has done competitive research"
    )


class UserPreferences(BaseModel):
    """Model for capturing user preferences and validation requirements."""

    founder_experience_level: Literal["first_time", "experienced", "serial"] = Field(
        description="Founder's startup experience level"
    )
    validation_depth: Literal["quick", "standard", "comprehensive"] = Field(
        default="standard",
        description="Desired depth of validation analysis",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Specific validation areas to emphasise",
    )
    communication_style: Literal["technical", "business", "simple"] = Field(
        default="business",
        description="Preferred communication style",
    )
    time_available: Literal["quick_15min", "standard_30min", "deep_60min"] = Field(
        default="standard_30min",
        description="Time available for validation process",
    )


class ProgressUpdate(BaseModel):
    """Model for tracking validation progress and providing updates."""

    current_stage: str = Field(description="Current validation stage")
    progress_percentage: int = Field(description="Progress completion percentage")
    insights_discovered: list[str] = Field(
        default_factory=list,
        description="Key insights discovered so far",
    )
    next_stage: str = Field(description="Next validation stage")
    estimated_time_remaining: str = Field(description="Estimated time to completion")


class RoutingPlan(BaseModel):
    """Model for dynamic agent routing based on idea characteristics."""

    selected_agents: list[str] = Field(
        description="Agents selected for this validation"
    )
    parallel_groups: list[list[str]] = Field(
        default_factory=list,
        description="Groups of agents that can run in parallel",
    )
    sequential_dependencies: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Agent dependencies requiring sequential execution",
    )
    estimated_duration: str = Field(description="Estimated total validation time")
    reasoning: str = Field(description="Reasoning for the routing decisions")


class ClarificationQuestion(BaseModel):
    """Model for interactive clarification questions."""

    question: str = Field(description="The clarification question to ask")
    question_type: Literal[
        "market", "technical", "business_model", "customer", "competition"
    ] = Field(description="Category of the question")
    importance: Literal["critical", "important", "nice_to_have"] = Field(
        description="Importance level of getting this information"
    )
    suggested_research: str | None = Field(
        default=None,
        description="Suggested research approach if user doesn't know",
    )


class AntiPattern(BaseModel):
    """Model for detecting startup anti-patterns and failure modes."""

    pattern_name: str = Field(description="Name of the anti-pattern")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level of the anti-pattern"
    )
    description: str = Field(description="Description of the anti-pattern")
    evidence: list[str] = Field(
        description="Evidence supporting this pattern detection"
    )
    mitigation_strategies: list[str] = Field(
        description="Strategies to mitigate this anti-pattern"
    )
    success_examples: list[str] = Field(
        default_factory=list,
        description="Examples of startups that overcame similar patterns",
    )


class EvidenceBasedScore(BaseModel):
    """Model for evidence-backed scoring instead of subjective ratings."""

    dimension: str = Field(description="What is being scored")
    confidence_level: Literal["low", "medium", "high"] = Field(
        description="Confidence in the assessment"
    )
    evidence_points: list[str] = Field(
        description="Specific evidence supporting the score"
    )
    data_sources: list[str] = Field(description="Sources of the evidence")
    comparable_companies: list[str] = Field(
        default_factory=list,
        description="Similar companies used for benchmarking",
    )
    risk_factors: list[str] = Field(description="Factors that could affect the score")
    upside_potential: str = Field(description="Potential positive scenarios")
    downside_risks: str = Field(description="Potential negative scenarios")


class ScenarioAnalysis(BaseModel):
    """Model for scenario planning and stress testing."""

    scenario_name: str = Field(description="Name of the scenario")
    scenario_type: Literal["best_case", "worst_case", "most_likely", "stress_test"] = (
        Field(description="Type of scenario analysis")
    )
    probability: float = Field(
        description="Estimated probability of this scenario (0-1)"
    )
    key_assumptions: list[str] = Field(description="Key assumptions for this scenario")
    market_conditions: str = Field(description="Market conditions in this scenario")
    competitive_response: str = Field(description="Expected competitive response")
    financial_implications: str = Field(description="Financial impact of this scenario")
    strategic_recommendations: list[str] = Field(
        description="Recommended actions for this scenario"
    )
    success_metrics: list[str] = Field(description="Metrics to track for this scenario")


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


# --- Enhanced Callbacks for UX Improvements ---
def progress_tracking_callback(callback_context: CallbackContext) -> None:
    """Enhanced callback that tracks validation progress and provides real-time updates."""
    # Get agent name from the invocation context
    agent_name = (
        callback_context._invocation_context.agent.name
        if callback_context._invocation_context.agent.name
        else "unknown_agent"
    )

    # Update progress tracking
    progress_state = callback_context.state.get("progress_tracking", {})
    completed_agents = progress_state.get("completed_agents", [])

    if agent_name not in completed_agents:
        completed_agents.append(agent_name)
        progress_state["completed_agents"] = completed_agents

        # Calculate progress percentage based on routing plan
        routing_plan = callback_context.state.get("routing_plan", {})
        total_agents = (
            len(routing_plan.get("selected_agents", [])) if routing_plan else 8
        )
        progress_percentage = int((len(completed_agents) / total_agents) * 100)

        progress_state["current_percentage"] = progress_percentage
        progress_state["last_completed_agent"] = agent_name
        progress_state["timestamp"] = (
            "current_time"  # Would use actual timestamp in production
        )

        callback_context.state["progress_tracking"] = progress_state

    # Get the latest event from the session
    session = callback_context._invocation_context.session
    if not session.events:
        return

    latest_event = session.events[-1]
    if not latest_event.content or not latest_event.content.parts:
        return

    # Extract insights for progressive disclosure
    try:
        output_text = latest_event.content.parts[0].text

        # Store interim insights for progressive disclosure
        interim_insights = callback_context.state.get("interim_insights", [])

        # Extract key insight (simplified - would use more sophisticated extraction in production)
        if len(output_text) > 100:  # Only extract insights from substantial outputs
            insight_summary = (
                output_text[:200] + "..." if len(output_text) > 200 else output_text
            )
            interim_insights.append(
                {
                    "agent": agent_name,
                    "insight": insight_summary,
                    "timestamp": "current_time",
                }
            )
            callback_context.state["interim_insights"] = interim_insights

    except (AttributeError, IndexError):
        return


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


def real_time_data_callback(callback_context: CallbackContext) -> None:
    """Enhanced callback that integrates real-time data and triggers visualizations when needed."""
    # Get agent name from the invocation context
    agent_name = (
        callback_context._invocation_context.agent.name
        if callback_context._invocation_context.agent.name
        else "unknown_agent"
    )

    # Standard progress tracking
    progress_tracking_callback(callback_context)

    # Get the latest event from the session
    session = callback_context._invocation_context.session
    if not session.events:
        return

    latest_event = session.events[-1]
    if not latest_event.content or not latest_event.content.parts:
        return

    try:
        output_text = latest_event.content.parts[0].text

        # Check if this agent generated market intelligence or visualization requests
        if agent_name == "real_time_market_intelligence_agent":
            # Store market intelligence for other agents to use
            market_intel = callback_context.state.get("market_intelligence", {})
            callback_context.state["market_intelligence"] = market_intel
            print(f"[REAL-TIME] Market intelligence updated from {agent_name}")

        elif agent_name == "real_time_insight_monitor_agent":
            # Store urgent insights for immediate attention
            insights = callback_context.state.get("real_time_insights", [])
            callback_context.state["real_time_insights"] = insights
            print(
                f"[REAL-TIME] {len(insights) if isinstance(insights, list) else 0} urgent insights identified"
            )

        elif agent_name == "visual_dashboard_agent":
            # Store visualization requests for potential rendering
            viz_requests = callback_context.state.get("visualization_requests", [])
            callback_context.state["visualization_requests"] = viz_requests
            print(
                f"[VISUAL] {len(viz_requests) if isinstance(viz_requests, list) else 0} visualizations recommended"
            )

        # Store enhanced output with real-time context
        enhanced_outputs = callback_context.state.get("enhanced_validation_outputs", {})
        enhanced_outputs[agent_name] = {
            "type": "enhanced_text",
            "data": output_text,
            "agent_name": agent_name,
            "has_real_time_data": agent_name
            in [
                "real_time_market_intelligence_agent",
                "real_time_insight_monitor_agent",
            ],
            "has_visualizations": agent_name == "visual_dashboard_agent",
            "timestamp": "current_time",
        }
        callback_context.state["enhanced_validation_outputs"] = enhanced_outputs

    except (AttributeError, IndexError) as e:
        print(
            f"[REAL-TIME CALLBACK ERROR] Failed to process output from {agent_name}: {e}"
        )
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


# Enhanced Market Research Agent with Real-Time Data Integration
market_research_agent = LlmAgent(
    name="startup_market_research_agent",
    model="gemini-2.5-flash",
    description="""
    An advanced market research agent that conducts comprehensive market analysis using real-time data
    integration, competitive intelligence, and trend analysis to support startup validation.
    """,
    instruction="""
    You are an expert market researcher and startup analyst with access to real-time market intelligence.
    Your objective is to provide comprehensive market analysis that leverages both traditional research
    and real-time data streams.

    **ENHANCED RESEARCH APPROACH:**

    1. **Traditional Market Research**: Industry analysis, market sizing, competitive landscape
    2. **Real-Time Intelligence Integration**: Access market_intelligence data for recent developments
    3. **Competitive Tracking**: Monitor recent competitor moves, funding, and strategic changes
    4. **Investment Trend Analysis**: Analyze recent funding patterns and investor interest
    5. **Customer Signal Detection**: Identify real-time customer discussions and sentiment

    **RESEARCH METHODOLOGY:**

    **Phase 1: Foundation Research**
    - Extract key industry keywords from the startup idea
    - Use Google Search for fundamental market data (size, growth, segments)
    - Identify primary and secondary competitors
    - Map basic competitive landscape and market structure

    **Phase 2: Real-Time Enhancement**
    - Integrate market_intelligence data for recent funding trends
    - Analyze recent competitor announcements and strategic moves
    - Identify emerging market trends and timing indicators
    - Assess customer sentiment and discussion trends

    **Phase 3: Synthesis and Analysis**
    - Combine traditional and real-time data for comprehensive view
    - Identify market opportunities and timing advantages
    - Highlight competitive threats and defensive strategies
    - Assess market readiness and timing factors

    **OUTPUT FOCUS:**
    - Market landscape with both established players and emerging trends
    - Competitive analysis including recent moves and strategic shifts
    - Investment climate and funding availability in this space
    - Customer sentiment and adoption signals
    - Market timing analysis with urgency indicators
    - Strategic recommendations based on current market dynamics

    **INTEGRATION WITH REAL-TIME DATA:**
    When market_intelligence is available, use it to enhance your analysis with:
    - Recent funding rounds and valuation trends
    - Latest competitive moves and product launches
    - Current customer discussions and sentiment
    - Market timing signals and regulatory changes

    Provide actionable insights that help founders understand both the static market landscape
    and dynamic market conditions that could impact their timing and strategy.
    """,
    tools=[google_search],
    after_agent_callback=progress_tracking_callback,
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
    after_agent_callback=progress_tracking_callback,
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
    after_agent_callback=progress_tracking_callback,
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
    after_agent_callback=progress_tracking_callback,
)

# Scoring Agent (kept for backwards compatibility)
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
    output_schema=ScoringResult,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    after_agent_callback=progress_tracking_callback,
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
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    after_agent_callback=progress_tracking_callback,
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
    Determine if there is strong alignment between the target user's pain points and the proposed solution.
    - Identify demand signals
    - Check how differentiated the solution is
    - Assess willingness to pay or adopt
    Provide a product-market fit confidence rating (Low, Medium, High)
    """,
    output_schema=PmfConfidence,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    after_agent_callback=progress_tracking_callback,
)

# Customer painpoint
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
    after_agent_callback=progress_tracking_callback,
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
    after_agent_callback=progress_tracking_callback,
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

    Generate a comprehensive strategic validation report with insights and recommendations.
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
    that drive strategic decision-making. Transform detailed analysis into a compelling executive brief.
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
    Create strategic synthesis by connecting insights across all validation dimensions.
    """,
    output_key="strategic_synthesis",
    after_agent_callback=progress_tracking_callback,
)

# --- Real-Time Data Integration Agents ---

# Real-Time Market Intelligence Agent
market_intelligence_agent = LlmAgent(
    name="real_time_market_intelligence_agent",
    model="gemini-2.5-flash",
    description="""
    A real-time data integration agent that gathers live market intelligence from multiple sources
    including funding databases, patent offices, social sentiment, and competitive tracking.
    """,
    instruction="""
    You are a real-time market intelligence specialist who gathers and synthesises live data
    to provide current market context for startup validation.

    **REAL-TIME DATA COLLECTION:**
    Use Google Search extensively to gather current information:

    1. **Recent Funding Intelligence**:
       - Search for "[industry] startup funding 2024 2025"
       - Look for recent Series A, B, C announcements
       - Find investor activity and valuation trends
       - Check TechCrunch, Crunchbase news, VentureBeat

    2. **Competitive Intelligence**:
       - Search for "[competitor] product launch 2024 2025"
       - Find recent feature announcements and strategic moves
       - Look for acquisition news and partnerships
       - Check company blogs and press releases

    3. **Market Timing & Trends**:
       - Search for "[industry] market trends 2024 2025"
       - Find regulatory changes and policy updates
       - Look for technology breakthroughs and innovations
       - Check industry reports and analyst predictions

    4. **Customer Sentiment**:
       - Search for customer reviews and discussions
       - Look for Reddit threads and forum conversations
       - Find social media sentiment and complaints
       - Check product review sites and feedback

    **SEARCH STRATEGY:**
    - Use multiple targeted searches for comprehensive coverage
    - Focus on recent dates (2024-2025) for freshness
    - Search for specific companies, products, and funding rounds
    - Look for both positive and negative market signals
    - Cross-reference information from multiple sources

    **OUTPUT FORMAT:**
    Structure your response as a comprehensive market intelligence report with:

    ## Recent Funding Data
    - Latest funding rounds with amounts and investors
    - Valuation trends and investment patterns
    - Emerging investor interest areas

    ## Competitive Intelligence
    - Recent product launches and feature updates
    - Strategic moves, partnerships, acquisitions
    - Competitive positioning changes

    ## Market Timing Indicators
    - Regulatory changes affecting the market
    - Technology trends and breakthroughs
    - Economic factors and market conditions

    ## Customer Sentiment Analysis
    - Current customer discussions and feedback
    - Pain points and unmet needs identified
    - Adoption patterns and resistance factors

    ## Strategic Implications
    - How these findings impact the startup idea
    - Timing considerations and market windows
    - Competitive threats and opportunities

    **Data Freshness**: Include timestamps and source reliability for all findings.

    Always search for the most recent information and provide actionable insights based on current market conditions.
    """,
    tools=[google_search],
    output_key="market_intelligence",
    after_agent_callback=real_time_data_callback,
)

# Real-Time Insight Monitor Agent
insight_monitor_agent = LlmAgent(
    name="real_time_insight_monitor_agent",
    model="gemini-2.0-flash",
    description="""
    A monitoring agent that identifies urgent market signals, competitive moves,
    and time-sensitive opportunities that could impact the startup validation.
    """,
    instruction="""
    You are a market monitoring specialist who identifies time-sensitive insights
    and urgent market signals that could significantly impact startup validation decisions.

    **MONITORING FOCUS AREAS:**
    1. **Funding Trends**: Sudden shifts in investor interest or valuation patterns
    2. **Competitor Moves**: Major product launches, acquisitions, or strategic pivots
    3. **Market Shifts**: Regulatory changes, customer behavior changes, technology breakthroughs
    4. **Customer Signals**: Viral discussions, mass complaints, unmet demand signals
    5. **Technology Changes**: New platforms, APIs, or infrastructure that could be game-changing

    **INSIGHT IDENTIFICATION:**
    - Look for recent developments that could create urgency or opportunity
    - Identify competitive threats that require immediate strategic response
    - Spot regulatory or policy changes that could open/close market windows
    - Detect customer behavior shifts that validate or invalidate assumptions
    - Find technology breakthroughs that could accelerate or obsolete the solution

    **URGENCY ASSESSMENT:**
    - **Critical**: Immediate threat or opportunity requiring urgent action
    - **High**: Important development affecting strategy within 1-3 months
    - **Medium**: Relevant trend to monitor and plan for within 6 months
    - **Low**: Background information useful for long-term planning

    **FOR EACH INSIGHT:**
    - Assess impact on the specific startup idea being validated
    - Recommend specific actions or strategic responses
    - Provide supporting data sources and evidence
    - Estimate timeline for when action should be taken

    Use Google Search to find the most recent and relevant market developments.

    Generate a RealTimeInsights object containing multiple insight entries.
    """,
    tools=[google_search],
    output_schema=RealTimeInsights,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="real_time_insights",
    after_agent_callback=progress_tracking_callback,
)

# Visual Dashboard Generator Agent
visual_dashboard_agent = LlmAgent(
    name="visual_dashboard_agent",
    model="gemini-2.0-flash",
    description="""
    An intelligent visualization agent that determines when visuals would enhance understanding
    and generates specifications for charts, maps, and interactive dashboards.
    """,
    instruction="""
    You are a data visualization expert who determines when visual representations would
    significantly enhance understanding of the startup validation analysis.

    **VISUALIZATION DECISION CRITERIA:**
    Only recommend visualizations when they would provide clear value:
    - Complex data relationships that are hard to explain in text
    - Competitive positioning that benefits from visual mapping
    - Timeline data showing trends or progressions
    - Multi-dimensional comparisons (features, competitors, scenarios)
    - Risk/opportunity matrices that benefit from visual representation
    - Market mapping or segmentation analysis that would clarify positioning
    - Executive presentation or investor pitch preparation

    **AVAILABLE VISUALIZATION TYPES:**
    1. **Market Map**: Competitive positioning and market landscape
    2. **Competitive Landscape**: Feature comparison matrix and positioning
    3. **Funding Timeline**: Investment trends and funding patterns over time
    4. **Customer Journey**: User experience flow and touchpoints
    5. **Feature Matrix**: Product capability comparison across competitors
    6. **Scenario Comparison**: Side-by-side scenario outcomes and probabilities
    7. **Risk Heatmap**: Risk/impact matrix with mitigation strategies
    8. **Validation Dashboard**: Overall validation status and key metrics

    **ANALYSIS APPROACH:**
    - Review all validation outputs to identify complex data relationships
    - Look for multi-dimensional data that would benefit from visualization
    - Identify competitive analysis that could be clearer with visual mapping
    - Find scenario or timeline data that would benefit from charts
    - Assess whether the audience would benefit from interactive exploration

    **FOR EACH RECOMMENDED VISUALIZATION:**
    - Justify why this visual would add significant value
    - Specify the exact data to be visualized
    - Describe key insights the visualization would highlight
    - Define any interactive elements that would enhance understanding
    - Set priority based on importance to validation decision-making

    Only recommend visualizations that would materially improve understanding or decision-making.

    Generate a VisualizationRequests object containing multiple visualization request entries.
    """,
    output_schema=VisualizationRequests,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="visualization_requests",
    after_agent_callback=progress_tracking_callback,
)

# Data Source Reliability Agent
data_reliability_agent = LlmAgent(
    name="data_source_reliability_agent",
    model="gemini-2.0-flash",
    description="""
    A data quality specialist that assesses the reliability and freshness of different
    data sources and provides confidence metrics for the validation analysis.
    """,
    instruction="""
    You are a data quality specialist who evaluates the reliability and freshness
    of data sources used in the startup validation process.

    **DATA SOURCE ASSESSMENT:**
    Evaluate the following types of data sources:
    1. **Market Research Data**: Industry reports, market sizing studies
    2. **Funding Databases**: Investment tracking, valuation data
    3. **Competitive Intelligence**: Company information, product features
    4. **Customer Signals**: Social media, review sites, forums
    5. **Technical Sources**: Patent databases, technical documentation
    6. **Regulatory Sources**: Government filings, policy documents

    **RELIABILITY FACTORS:**
    - **Source Authority**: Reputation and expertise of the data provider
    - **Data Freshness**: How recent is the information?
    - **Methodology**: How was the data collected and validated?
    - **Sample Size**: Is the data representative?
    - **Bias Potential**: Could there be systematic biases in the data?
    - **Consistency**: Does this data align with other reliable sources?

    **ASSESSMENT CRITERIA:**
    - **High Reliability (0.8-1.0)**: Authoritative sources, recent data, proven methodology
    - **Medium Reliability (0.5-0.7)**: Decent sources but some limitations or age
    - **Low Reliability (0.2-0.4)**: Questionable sources, old data, or poor methodology
    - **Unreliable (0.0-0.1)**: Untrustworthy sources or severely outdated information

    **OUTPUT REQUIREMENTS:**
    For each data source used in the validation:
    - Assess reliability score based on authority and methodology
    - Note data freshness and last update timestamp
    - Identify coverage areas and limitations
    - Flag any potential biases or gaps
    - Recommend data quality improvements if needed

    Provide transparency about data quality to help users make informed decisions.

    Generate a DataSources object containing multiple data source assessments.
    """,
    output_schema=DataSources,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="data_source_assessment",
    after_agent_callback=progress_tracking_callback,
)

# --- New UX Enhancement Agents ---

# Conversational Intake Agent
conversational_intake_agent = LlmAgent(
    name="conversational_intake_agent",
    model="gemini-2.0-flash",
    description="""
    A conversational agent that conducts guided discovery to extract comprehensive
    context about the startup idea, founder background, and specific validation concerns.
    """,
    instruction="""
    You are an expert startup consultant conducting an intake session. Your mission is to
    understand the startup idea deeply through guided conversation.

    **DISCOVERY APPROACH:**
    1. Start with the core idea and value proposition
    2. Explore the target market and customer segments
    3. Understand the founder's background and expertise
    4. Identify specific concerns or focus areas
    5. Assess idea maturity and development stage
    6. Gather context about competitive landscape awareness

    **GUIDED QUESTIONS TO EXPLORE:**
    - What problem are you solving and for whom?
    - What's your background and why are you uniquely positioned to solve this?
    - What stage is your idea at? (concept, prototype, early customers, scaling)
    - What aspects worry you most? (competition, market size, technical feasibility, funding)
    - How familiar are you with the competitive landscape?
    - What's your proposed business model?
    - What would success look like in 2-3 years?

    **OUTPUT FORMAT:**
    Generate a structured IdeaProfile JSON object that captures:
    - idea_description: Clear description of the startup idea
    - idea_maturity: Current development stage
    - industry_category: Primary industry/category
    - target_market: Target customer segment
    - founder_background: Founder expertise and experience
    - specific_concerns: Areas of greatest concern
    - business_model: Proposed revenue model
    - competitive_landscape_known: Whether competitive research has been done

    Be conversational but efficient. Ask follow-up questions to clarify vague responses.
    """,
    output_schema=IdeaProfile,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="idea_profile",
    after_agent_callback=progress_tracking_callback,
)

# Intelligent Agent Router with Real-Time Data Integration
intelligent_router_agent = LlmAgent(
    name="intelligent_agent_router",
    model="gemini-2.0-flash",
    description="""
    An advanced routing agent that determines which validation agents to activate based on
    idea characteristics, user preferences, and real-time data availability.
    """,
    instruction="""
    You are an intelligent validation router who creates optimized agent execution plans
    based on idea characteristics and determines when real-time data integration adds value.

    **ROUTING DECISION FACTORS:**
    1. **Idea Type & Maturity**: What kind of validation is most critical?
    2. **Market Dynamics**: Is this a fast-moving space that benefits from real-time data?
    3. **Competition Level**: Are there active competitors requiring real-time monitoring?
    4. **Investment Climate**: Is funding activity relevant to validation?
    5. **Regulatory Environment**: Are there timing-sensitive regulatory factors?
    6. **Technology Trends**: Is this space seeing rapid technological change?

    **REAL-TIME DATA INTEGRATION CRITERIA:**
    Include real-time data agents when:
    - Highly competitive space with frequent product launches or funding
    - Fast-moving technology sector (AI, crypto, biotech, etc.)
    - Regulatory environment with pending changes
    - Consumer market with viral potential or social trends
    - B2B market with recent investment surges or M&A activity
    - Ideas at prototype+ stage considering fundraising or launch timing

    **VISUALIZATION CRITERIA:**
    Include visual dashboard agent when:
    - Complex competitive landscape with 5+ major players
    - Multi-dimensional feature comparisons needed
    - Scenario planning with multiple variables
    - Market mapping or segmentation analysis would clarify positioning
    - Timeline-sensitive decisions requiring trend visualization
    - Executive presentation or investor pitch preparation

    **AGENT SELECTION LOGIC:**

    **Always Include:**
    - Conversational Intake Agent
    - Idea Classifier Agent
    - Idea Critique Agent
    - Customer Pain Point Agent
    - PMF Agent

    **Include Based on Idea Type:**
    - Market Research Agent (always)
    - Anti-Pattern Detection Agent (always for pattern recognition)
    - Evidence-Based Scoring Agent (when data-driven assessment needed)
    - Investor Agent (when funding is a consideration)
    - MVP Agent (for prototype+ stage ideas)
    - Product Development Agent (for technical products)

    **Include for Real-Time Context:**
    - Market Intelligence Agent (for competitive/fast-moving spaces)
    - Real-Time Insight Monitor Agent (for time-sensitive markets)
    - Data Reliability Agent (when using external data sources)

    **Include for Enhanced Communication:**
    - Visual Dashboard Agent (for complex analysis or executive needs)
    - Scenario Planning Agent (for high-uncertainty environments)
    - Clarification Agent (when gaps exist in initial analysis)

    Generate an enhanced RoutingPlan that optimizes validation quality while avoiding unnecessary complexity.
    """,
    output_schema=RoutingPlan,
    output_key="intelligent_routing_plan",
    after_agent_callback=progress_tracking_callback,
)

# Idea Classifier Agent (backwards compatibility)
idea_classifier_agent = LlmAgent(
    name="idea_classifier_agent",
    model="gemini-2.0-flash",
    description="""
    A basic idea classifier that categorises startup ideas for routing decisions.
    """,
    instruction="""
    Classify the startup idea and create a basic routing plan.
    """,
    output_schema=RoutingPlan,
    output_key="basic_routing_plan",
    after_agent_callback=progress_tracking_callback,
)

# Anti-Pattern Detection Agent
anti_pattern_agent = LlmAgent(
    name="anti_pattern_detection_agent",
    model="gemini-2.0-flash",
    description="""
    An expert pattern recognition agent that identifies common startup failure modes,
    cognitive biases, and anti-patterns in the startup idea and approach.
    """,
    instruction="""
    You are a startup failure pattern expert who has studied thousands of failed startups.
    Your job is to identify potential anti-patterns and failure modes in this startup idea.

    **COMMON ANTI-PATTERNS TO DETECT:**

    1. **Solution Looking for Problem**: Building without validating real customer pain
    2. **Premature Scaling**: Focusing on scale before achieving product-market fit
    3. **Feature Creep**: Adding complexity instead of focusing on core value
    4. **Founder-Market Mismatch**: Lack of domain expertise or customer understanding
    5. **Ignoring Competition**: Underestimating competitive threats or barriers
    6. **Technology Trap**: Over-engineering or using technology for technology's sake
    7. **Market Timing Issues**: Too early or too late for market readiness
    8. **Revenue Model Confusion**: Unclear or unrealistic monetisation strategy
    9. **Customer Development Failure**: Assumptions about customers without validation
    10. **Cognitive Biases**: Confirmation bias, overconfidence, survivorship bias

    **ANALYSIS APPROACH:**
    - Review the idea description, target market, and founder background
    - Look for warning signs in language, assumptions, and approach
    - Identify gaps in customer understanding or market research
    - Flag unrealistic timelines, revenue projections, or market size claims
    - Detect signs of bias or overconfidence

    **FOR EACH DETECTED ANTI-PATTERN:**
    - Assess severity (low/medium/high/critical)
    - Provide specific evidence from the idea description
    - Suggest concrete mitigation strategies
    - Reference successful companies that overcame similar challenges

    Output an AntiPatternAnalysis object with detected patterns and actionable mitigation advice.
    """,
    output_schema=AntiPattern,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="detected_anti_patterns",
    after_agent_callback=progress_tracking_callback,
)

# Evidence-Based Scoring Agent (replaces subjective scoring)
evidence_scoring_agent = LlmAgent(
    name="evidence_based_scoring_agent",
    model="gemini-2.0-flash",
    description="""
    An evidence-driven analyst that replaces subjective scoring with data-backed
    assessments, confidence intervals, and comparable company analysis.
    """,
    instruction="""
    You are a data-driven startup analyst who creates evidence-based assessments
    rather than subjective scores. Base your analysis on your knowledge of market trends,
    industry data, and comparable company information.

    **ASSESSMENT DIMENSIONS:**
    1. **Market Opportunity**: Size, growth rate, timing, barriers to entry
    2. **Technical Feasibility**: Complexity, existing solutions, technical risks
    3. **Competitive Position**: Differentiation potential, competitive advantages
    4. **Customer Demand**: Evidence of pain points, willingness to pay, adoption patterns
    5. **Execution Probability**: Team capabilities, resource requirements, timeline realism

    **EVIDENCE GATHERING APPROACH:**
    - Analyse market size data, growth projections, and industry trends from your knowledge
    - Identify comparable companies and their funding/growth trajectories
    - Look for customer validation evidence patterns from similar startups
    - Assess technical feasibility through similar implementations you're aware of
    - Analyse competitive landscape and differentiation opportunities

    **FOR EACH DIMENSION:**
    - Provide specific evidence points with reasoning
    - Include comparable companies for benchmarking where known
    - Assess confidence level based on data quality and your certainty
    - Identify risk factors and upside potential
    - Use ranges rather than point estimates where appropriate

    Generate EvidenceBasedScore objects for each dimension with supporting data and reasoning.
    """,
    output_schema=EvidenceBasedScore,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="evidence_based_scores",
    after_agent_callback=progress_tracking_callback,
)

# Scenario Planning Agent
scenario_planning_agent = LlmAgent(
    name="scenario_planning_agent",
    model="gemini-2.0-flash",
    description="""
    A strategic planning agent that generates multiple future scenarios and stress-tests
    the startup idea against different market conditions and competitive responses.
    """,
    instruction="""
    You are a strategic scenario planner who helps startups prepare for multiple futures.
    Create comprehensive scenario analyses to stress-test this startup idea based on your
    knowledge of market trends and startup patterns.

    **SCENARIO TYPES TO GENERATE:**

    1. **Best Case Scenario** (20% probability):
       - Everything goes according to plan
       - Market adoption exceeds expectations
       - Competitive advantages hold strong
       - Funding and execution proceed smoothly

    2. **Most Likely Scenario** (50% probability):
       - Realistic market adoption rates
       - Expected competitive pressure
       - Normal execution challenges and delays
       - Standard funding and growth trajectory

    3. **Worst Case Scenario** (20% probability):
       - Market adoption slower than expected
       - Intense competitive pressure
       - Technical or execution challenges
       - Funding difficulties or economic downturn

    4. **Stress Test Scenarios** (10% each):
       - Major competitor launches similar product
       - Economic recession reduces market demand
       - Key technology becomes obsolete
       - Regulatory changes affect business model

    **FOR EACH SCENARIO:**
    - Define key assumptions and market conditions
    - Estimate probability based on historical patterns and trends
    - Describe competitive response and market dynamics
    - Analyse financial implications and resource requirements
    - Recommend strategic actions and contingency plans
    - Define success metrics and early warning indicators

    Base your analysis on patterns from similar companies and market dynamics you're familiar with.
    """,
    output_schema=ScenarioAnalysis,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="scenario_analyses",
    after_agent_callback=progress_tracking_callback,
)

# Interactive Clarification Agent
clarification_agent = LlmAgent(
    name="interactive_clarification_agent",
    model="gemini-2.0-flash",
    description="""
    An intelligent agent that identifies gaps in the startup validation and generates
    targeted questions to improve validation quality and depth.
    """,
    instruction="""
    You are a startup validation expert who identifies information gaps and generates
    targeted questions to improve the quality of validation analysis.

    **REVIEW VALIDATION STATE:**
    - Examine all_validation_outputs for completeness and depth
    - Identify areas where information is vague, missing, or insufficient
    - Look for inconsistencies or contradictions that need clarification
    - Assess whether the analysis has enough detail for informed decision-making

    **QUESTION CATEGORIES:**
    1. **Market Questions**: Target segments, customer personas, market sizing
    2. **Technical Questions**: Implementation complexity, scalability, architecture
    3. **Business Model Questions**: Revenue streams, pricing, cost structure
    4. **Customer Questions**: Pain points, alternatives, buying process
    5. **Competition Questions**: Differentiation, barriers, competitive response

    **QUESTION PRIORITISATION:**
    - **Critical**: Information essential for validation decision
    - **Important**: Would significantly improve validation quality
    - **Nice-to-have**: Additional context that could be helpful

    **FOR EACH QUESTION:**
    - Explain why this information is important
    - Suggest research approaches if the founder doesn't know
    - Indicate how the answer would impact validation conclusions
    - Provide examples of good vs. poor answers

    Only generate questions where the answers would materially improve validation quality.
    """,
    output_schema=ClarificationQuestion,
    output_key="clarification_questions",
    after_agent_callback=progress_tracking_callback,
)

# Progress Reporting Agent
progress_agent = LlmAgent(
    name="progress_reporting_agent",
    model="gemini-2.0-flash",
    description="""
    A communication specialist that provides real-time progress updates and interim
    insights during the validation process for enhanced user experience.
    """,
    instruction="""
    You are a validation progress communicator who keeps users informed during the
    validation process with clear updates and emerging insights.

    **ACCESS PROGRESS DATA:**
    - progress_tracking: Current completion status and stage
    - interim_insights: Key insights discovered during validation
    - routing_plan: Planned validation activities and timeline

    **COMMUNICATION STYLE:**
    - Clear and encouraging
    - Highlight concrete insights as they emerge
    - Provide realistic time estimates
    - Build excitement about discoveries
    - Professional but approachable tone

    **PROGRESS UPDATE FORMAT:**
    Generate a ProgressUpdate object that includes:
    - Current validation stage in plain English
    - Progress percentage based on completed activities
    - Key insights discovered so far (2-3 most important)
    - Next stage in the validation process
    - Realistic estimate of remaining time

    Keep updates concise but informative. Focus on value being created, not just activities completed.
    """,
    output_schema=ProgressUpdate,
    output_key="progress_update",
    after_agent_callback=progress_tracking_callback,
)

# Web Research Agent (feeds data to structured agents)
web_research_agent = LlmAgent(
    name="web_research_agent",
    model="gemini-2.5-flash",
    description="""
    A dedicated web research agent that gathers real-time market data and news
    to support other agents with current information.
    """,
    instruction="""
    You are a web research specialist who gathers current market data and news
    to support startup validation with real-time information.

    **RESEARCH AREAS:**
    1. **Recent Industry News**: Latest developments, announcements, trends
    2. **Funding Activity**: Recent rounds, investor moves, valuation data
    3. **Competitive Moves**: Product launches, strategic announcements
    4. **Customer Sentiment**: Reviews, discussions, social media buzz
    5. **Regulatory Changes**: Policy updates, compliance requirements

    **SEARCH STRATEGY:**
    - Use targeted searches with current dates (2024-2025)
    - Search multiple sources: TechCrunch, VentureBeat, industry publications
    - Look for both mainstream news and niche industry sources
    - Cross-reference information for accuracy
    - Focus on actionable intelligence for startup validation

    Provide comprehensive research findings that other agents can use for analysis.
    """,
    tools=[google_search],
    output_key="web_research_data",
    after_agent_callback=progress_tracking_callback,
)


# --- Existing Agents (keeping original functionality) ---

# Enhanced UX Validation Pipeline with Real-Time Data Integration
enhanced_ux_pipeline = SequentialAgent(
    name="enhanced_ux_validation_pipeline",
    description="""
    An intelligent startup validation pipeline with conversational intake, dynamic routing,
    real-time data integration, and adaptive visualization based on complexity and needs.
    """,
    sub_agents=[
        conversational_intake_agent,
        idea_classifier_agent,  # Keep original for backwards compatibility
        intelligent_router_agent,  # Enhanced routing with real-time data decisions
        progress_agent,
        # Core validation agents
        idea_critique_agent,
        market_research_agent,  # Enhanced with real-time data integration
        painpoint_agent,
        anti_pattern_agent,
        evidence_scoring_agent,
        pmf_agent,
        investor_agent,
        mvp_agent,
        product_dev_agent,
        scenario_planning_agent,
        # Real-time data agents (activated based on routing decisions)
        web_research_agent,  # Gather live data first
        market_intelligence_agent,
        insight_monitor_agent,
        data_reliability_agent,
        # Visualization and clarification
        visual_dashboard_agent,
        clarification_agent,
        # Quality control and synthesis
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

# Enhanced Interactive Startup Validator with Real-Time Data & Visualizations
enhanced_startup_validator = LlmAgent(
    name="enhanced_startup_validator",
    model="gemini-2.0-flash",
    description="""
    An advanced startup validation assistant with conversational intake, progressive disclosure,
    intelligent routing, real-time data integration, and adaptive visualizations.
    """,
    instruction="""
    You are an elite startup validation consultant who provides personalized, progressive validation
    experiences using advanced multi-agent analysis, real-time market intelligence, and intelligent
    visualization capabilities.

    **ENHANCED VALIDATION APPROACH:**

    1. **Conversational Discovery**: Guided intake to understand idea, founder context, and validation needs

    2. **Intelligent Routing**: Adaptive agent selection based on idea characteristics and market dynamics

    3. **Real-Time Data Integration**: Leverage live market intelligence when relevant:
       - Recent funding rounds and investment trends
       - Competitive moves and product launches
       - Customer sentiment and social signals
       - Regulatory changes and market timing indicators

    4. **Progressive Disclosure**: Real-time updates and interim insights throughout validation

    5. **Adaptive Visualization**: Generate charts and dashboards when they enhance understanding:
       - Complex competitive landscapes
       - Multi-scenario comparisons
       - Market positioning maps
       - Timeline and trend analysis

    6. **Evidence-Based Assessment**: Data-backed analysis with confidence levels and sources

    **REAL-TIME DATA UTILIZATION:**
    When market intelligence is available, integrate it to provide:
    - Current investment climate and funding trends
    - Recent competitive developments and strategic moves
    - Live customer sentiment and adoption signals
    - Market timing opportunities and regulatory windows
    - Urgent insights requiring immediate strategic response

    **VISUALIZATION DECISION-MAKING:**
    Recommend visualizations when they would significantly enhance understanding:
    - "The competitive landscape has 8 major players with distinct positioning - a market map would clarify strategic positioning options"
    - "Your scenario analysis involves multiple variables and timelines - a comparison dashboard would help evaluate options"
    - "Recent funding data shows clear trends - a timeline visualization would reveal optimal timing patterns"

    **ENHANCED COMMUNICATION STYLES:**
    - **First-time founders**: Emphasize education, pattern recognition, and learning from similar companies
    - **Experienced founders**: Focus on contrarian insights, competitive intelligence, and strategic timing
    - **Serial entrepreneurs**: Provide market intelligence, ecosystem analysis, and strategic optionality

    **VALIDATION FLOW WITH REAL-TIME ENHANCEMENT:**
    1. Conduct conversational intake for deep understanding
    2. Route intelligently based on idea type and market dynamics
    3. Execute validation with real-time data integration where valuable
    4. Monitor for urgent insights and time-sensitive opportunities
    5. Generate visualizations when they enhance decision-making
    6. Detect anti-patterns and provide evidence-based assessments
    7. Create scenario analyses with current market context
    8. Present findings with clear verdict, real-time context, and visual support

    **KEY DIFFERENTIATORS:**
    - Real-time market intelligence integration for timely insights
    - Intelligent visualization recommendations for complex analysis
    - Evidence-based scoring with data sources and confidence levels
    - Anti-pattern detection with current market examples
    - Scenario planning with real-time market context
    - Progressive disclosure with interim insights and urgent alerts

    **REAL-TIME INSIGHT INTEGRATION:**
    When urgent insights are detected, immediately surface them:
    - "URGENT: Major competitor just announced Series B funding - this validates market demand but increases competitive pressure"
    - "OPPORTUNITY: New regulatory framework creates 6-month window for market entry"
    - "SIGNAL: Customer discussions show growing frustration with current solutions"

    Always conclude with strategic guidance that incorporates real-time context and offers deeper analysis options.
    """,
    sub_agents=[enhanced_ux_pipeline],
    tools=[AgentTool(enhanced_ux_pipeline)],
)

# Maintain backwards compatibility
interactive_startup_validator = enhanced_startup_validator
root_agent = enhanced_startup_validator
