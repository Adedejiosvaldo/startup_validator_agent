"""Turns the structured session state into the final markdown report.

Kept free of ADK imports so it can be unit-tested without the SDK or an API key.
No LLM call here: the report only shows what the agents produced.
"""

from typing import Any

CHANNEL_LABELS = {
    "warm_outreach": "Warm outreach (people who already know you)",
    "cold_outreach": "Cold outreach",
    "content": "Free content",
    "paid_ads": "Paid ads",
}


def value_score(ve: dict[str, Any]) -> float | None:
    """Hormozi's Value Equation: (dream x likelihood) / (time delay x effort)."""
    try:
        top = ve["dream_outcome"] * ve["perceived_likelihood"]
        bottom = ve["time_delay"] * ve["effort_and_sacrifice"]
    except (KeyError, TypeError):
        return None
    return round(top / bottom, 2) if bottom else None


def _value_label(score: float | None) -> str:
    if score is None:
        return "n/a"
    if score >= 4:
        return f"{score} (strong)"
    if score >= 1:
        return f"{score} (okay)"
    return f"{score} (weak: costs the buyer more than it gives)"


def _confidence_line(verdict: dict[str, Any]) -> str:
    votes, confidence = verdict.get("votes"), verdict.get("confidence")
    if not votes:
        return ""
    if confidence == "high":
        return f"Confidence: **high**, all {votes} verdict runs agreed."
    if confidence == "split":
        return (
            f"Confidence: **split** ({votes}). The runs couldn't agree, so it defaults "
            "to KILL. Treat this as a coin flip."
        )
    return (
        f"Confidence: **low**, only {votes} runs agreed. It's a judgement call; "
        "the case against is worth reading."
    )


def _bullets(items: list[str] | None) -> str:
    return "\n".join(f"- {i}" for i in items or []) or "- none"


def render_report(state: dict[str, Any]) -> str:
    brief = state.get("idea_brief") or {}
    hormozi = state.get("hormozi") or {}
    africa = state.get("africa_lens") or {}
    verdict = state.get("verdict") or {}
    research = state.get("market_research") or ""

    if not verdict:
        return "⚠️ The verdict step didn't produce a result. Check the agent logs and run again."

    ve = hormozi.get("value_equation") or {}
    decision = verdict.get("decision", "?")
    icon = "🛑" if decision == "KILL" else "🧪"

    parts = [
        f"# {icon} {decision}: {brief.get('one_liner', 'Your idea')}",
        f"**{verdict.get('reason', '')}**",
    ]
    confidence = _confidence_line(verdict)
    if confidence:
        parts.append(confidence)
    parts += ["### Why", _bullets(verdict.get("evidence"))]

    test = verdict.get("test")
    if decision == "TEST" and test:
        parts += [
            f"### Your {test.get('days', 7)}-day test",
            f"- **Who:** {test.get('buyer')}",
            f"- **Where:** {test.get('where_to_find_them')}",
            f"- **Offer:** {test.get('offer')}",
            f"- **Price:** ₦{test.get('price_ngn', 0):,}",
            f"- **Pitch:** {test.get('people_to_contact')} people",
            f"- **Pass if:** {test.get('pass_threshold')}",
            "",
            "**First message (copy and send):**",
            "",
            f"> {test.get('first_message', '').strip()}".replace("\n", "\n> "),
        ]

    label = "Stop if" if decision == "TEST" else "What would revive it"
    parts += [f"**{label}:** {verdict.get('what_would_change_my_mind', '')}"]

    run_id = state.get("run_id")
    if run_id and decision == "TEST":
        parts += [
            f"**After the test, record what happened** (run `{run_id}`):",
            f"`python -m validator_v2.outcomes record {run_id} --result pass "
            f"--paid 3 --contacted 20 --notes \"...\"`",
        ]
    elif run_id:
        parts.append(f"_Logged as run `{run_id}`._")
    parts.append("---")

    bear, bull = state.get("bear_case"), state.get("bull_case")
    if bear or bull:
        parts += [
            "### Case to kill",
            str(bear or "n/a").strip(),
            "### Case to test",
            str(bull or "n/a").strip(),
        ]

    parts += [
        "### Value Equation",
        "| Dream outcome | Likelihood | Time delay | Effort | Value |",
        "|:-:|:-:|:-:|:-:|:-:|",
        f"| {ve.get('dream_outcome', '?')} | {ve.get('perceived_likelihood', '?')} | "
        f"{ve.get('time_delay', '?')} | {ve.get('effort_and_sacrifice', '?')} | "
        f"{_value_label(value_score(ve))} |",
        f"Weakest link: {ve.get('weakest_term', 'n/a')}",
        "### Offer",
        f"- **Grand Slam Offer:** {hormozi.get('grand_slam_offer', 'n/a')}",
        f"- **Guarantee:** {hormozi.get('guarantee', 'n/a')}",
        f"- **First cash offer:** {hormozi.get('attraction_offer', 'n/a')}",
        f"- **Start with:** "
        f"{CHANNEL_LABELS.get(hormozi.get('best_lead_channel', ''), 'n/a')}: "
        f"{hormozi.get('channel_reason', '')}",
        "### Africa lens",
        f"- **Trust:** {africa.get('trust_barrier', 'n/a')}",
        f"- **Distribution:** {africa.get('distribution', 'n/a')}",
        f"- **Payments:** {africa.get('payments', 'n/a')}",
        f"- **Missing infrastructure:** {africa.get('missing_infrastructure', 'n/a')}",
        f"- **Regulation:** {africa.get('regulation', 'n/a')}",
        f"- **Biggest blocker ({africa.get('blocker_severity', '?')}):** "
        f"{africa.get('hardest_blocker', 'n/a')}",
    ]

    if research:
        parts += ["### Market research", str(research).strip()]

    assumptions = brief.get("assumptions")
    if assumptions:
        parts += [
            "### Assumptions I made",
            "Correct any of these and run it again if they're wrong:",
            _bullets(assumptions),
        ]

    return "\n\n".join(parts)
