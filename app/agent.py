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
    output_key="idea_profile",
    after_agent_callback=progress_tracking_callback,
)

# Idea Classifier and Router Agent
idea_classifier_agent = LlmAgent(
    name="idea_classifier_agent",
    model="gemini-2.0-flash",
    description="""
    An intelligent classifier that categorises startup ideas and creates optimised
    validation routing plans based on idea characteristics and user preferences.
    """,
    instruction="""
    You are a startup validation strategist. Based on the idea_profile, create an optimised
    validation plan that adapts to the specific characteristics of this startup idea.

    **ANALYSIS FACTORS:**
    - Idea maturity stage (concept vs. prototype vs. early traction)
    - Industry type (B2B SaaS, consumer, marketplace, hardware, etc.)
    - Complexity level (simple app vs. complex platform vs. deep tech)
    - Market dynamics (established vs. emerging vs. creating new category)
    - Founder experience level and specific concerns

    **ROUTING STRATEGY:**
    - Select most relevant agents for this specific idea type
    - Identify agents that can run in parallel vs. those requiring sequential execution
    - Estimate time requirements based on complexity and depth needed
    - Prioritise agents addressing the founder's specific concerns

    **AGENT SELECTION LOGIC:**
    - Market Research: Always include for competitive landscape
    - Customer Pain Points: Critical for all B2C and most B2B ideas
    - PMF Analysis: Essential for all ideas
    - Investor Analysis: Include unless very early concept stage
    - MVP Planning: Include for prototype+ stage ideas
    - Technical Feasibility: Include for deep tech or complex platform ideas
    - Anti-pattern Detection: Always include for pattern recognition
    - Scenario Planning: Include for complex/high-risk ideas

    Generate a RoutingPlan that optimises the validation process for this specific idea.
    """,
    output_schema=RoutingPlan,
    output_key="routing_plan",
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

    Output a list of AntiPattern objects with actionable mitigation advice.
    """,
    tools=[google_search],
    output_schema=list[AntiPattern],
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
    rather than subjective scores. Use the Google Search tool to gather supporting data.

    **ASSESSMENT DIMENSIONS:**
    1. **Market Opportunity**: Size, growth rate, timing, barriers to entry
    2. **Technical Feasibility**: Complexity, existing solutions, technical risks
    3. **Competitive Position**: Differentiation potential, competitive advantages
    4. **Customer Demand**: Evidence of pain points, willingness to pay, adoption patterns
    5. **Execution Probability**: Team capabilities, resource requirements, timeline realism

    **EVIDENCE GATHERING APPROACH:**
    - Search for market size data, growth projections, and industry reports
    - Find comparable companies and their funding/growth trajectories
    - Look for customer validation evidence (surveys, pilot programs, early adoption)
    - Research technical feasibility through similar implementations
    - Analyze competitive landscape and differentiation opportunities

    **FOR EACH DIMENSION:**
    - Provide specific evidence points with sources
    - Include comparable companies for benchmarking
    - Assess confidence level based on data quality and quantity
    - Identify risk factors and upside potential
    - Use ranges rather than point estimates where appropriate

    Generate EvidenceBasedScore objects for each dimension with supporting data and sources.
    """,
    tools=[google_search],
    output_schema=list[EvidenceBasedScore],
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
    Create comprehensive scenario analyses to stress-test this startup idea.

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
    - Estimate probability based on historical data and trends
    - Describe competitive response and market dynamics
    - Analyze financial implications and resource requirements
    - Recommend strategic actions and contingency plans
    - Define success metrics and early warning indicators

    Use Google Search to gather data on similar companies' trajectories and market trends.
    """,
    tools=[google_search],
    output_schema=list[ScenarioAnalysis],
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
    output_schema=list[ClarificationQuestion],
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


# --- Existing Agents (keeping original functionality) ---

# Enhanced UX Validation Pipeline with Intelligent Routing
enhanced_ux_pipeline = SequentialAgent(
    name="enhanced_ux_validation_pipeline",
    description="""
    An intelligent startup validation pipeline with conversational intake, dynamic routing,
    progress tracking, and adaptive validation based on idea characteristics and user preferences.
    """,
    sub_agents=[
        conversational_intake_agent,
        idea_classifier_agent,
        progress_agent,
        # Core validation agents with enhanced UX
        idea_critique_agent,
        market_research_agent,
        painpoint_agent,
        anti_pattern_agent,
        evidence_scoring_agent,
        pmf_agent,
        investor_agent,
        mvp_agent,
        product_dev_agent,
        scenario_planning_agent,
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

# Enhanced Interactive Startup Validator with UX Improvements
enhanced_startup_validator = LlmAgent(
    name="enhanced_startup_validator",
    model="gemini-2.0-flash",
    description="""
    An advanced startup validation assistant with conversational intake, progressive disclosure,
    intelligent routing, and personalized validation experiences.
    """,
    instruction="""
    You are an elite startup validation consultant who provides personalized, progressive validation
    experiences using advanced multi-agent analysis and intelligent user experience design.

    **ENHANCED VALIDATION APPROACH:**

    1. **Conversational Discovery**: Begin with guided intake to understand the idea, founder context,
       and specific validation needs. Adapt your approach based on founder experience level.

    2. **Intelligent Routing**: Use the idea classification to create an optimized validation plan
       that focuses on the most relevant analyses for this specific idea type.

    3. **Progressive Disclosure**: Provide real-time updates and interim insights as validation
       progresses, rather than waiting until the end to share findings.

    4. **Adaptive Depth**: Adjust validation depth based on idea maturity, complexity, and
       available time. Focus on addressing the founder's specific concerns.

    5. **Interactive Clarification**: Ask targeted questions when additional information would
       significantly improve validation quality.

    **COMMUNICATION STYLE ADAPTATION:**
    - **First-time founders**: Use simpler language, more explanation, encourage exploration
    - **Experienced founders**: Focus on strategic insights, challenge assumptions, highlight nuances
    - **Serial entrepreneurs**: Provide contrarian perspectives, focus on differentiation and timing

    **VALIDATION FLOW:**
    1. Conduct conversational intake to understand the opportunity deeply
    2. Create intelligent routing plan based on idea characteristics
    3. Execute validation with progress updates and interim insights
    4. Identify and ask clarifying questions to improve analysis quality
    5. Detect anti-patterns and provide evidence-based assessments
    6. Generate scenario analyses and strategic recommendations
    7. Present findings with clear verdict and actionable next steps

    **KEY DIFFERENTIATORS:**
    - Personalized validation path based on idea and founder profile
    - Real-time insights during validation process
    - Evidence-backed assessments rather than subjective scores
    - Anti-pattern detection and mitigation strategies
    - Multiple scenario planning and stress testing
    - Interactive clarification for higher quality analysis

    Always conclude with clear strategic guidance and offer to dive deeper into specific areas.
    """,
    sub_agents=[enhanced_ux_pipeline],
    tools=[AgentTool(enhanced_ux_pipeline)],
)

# Maintain backwards compatibility
interactive_startup_validator = enhanced_startup_validator
root_agent = enhanced_startup_validator
