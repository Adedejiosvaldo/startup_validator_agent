"""Tests for the v2 report renderer. No ADK or API key needed:

    python -m pytest tests/
"""

import importlib.util
from pathlib import Path

# Load report.py directly so the test doesn't import the ADK via the package __init__.
_spec = importlib.util.spec_from_file_location(
    "report", Path(__file__).parent.parent / "validator_v2" / "report.py"
)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

BASE = {
    "idea_brief": {
        "one_liner": "Website scanner for NDPA compliance",
        "assumptions": ["Buyer is a Lagos fintech"],
    },
    "market_research": "**Competitors**: none local.",
    "hormozi": {
        "value_equation": {
            "dream_outcome": 8,
            "perceived_likelihood": 6,
            "time_delay": 2,
            "effort_and_sacrifice": 3,
            "weakest_term": "likelihood",
        },
        "grand_slam_offer": "Audit in 48h or it's free",
        "guarantee": "Money back",
        "attraction_offer": "N25k one-off audit",
        "best_lead_channel": "cold_outreach",
        "channel_reason": "No audience yet",
    },
    "africa_lens": {"blocker_severity": "manageable", "hardest_blocker": "trust"},
}


def test_value_score():
    assert report.value_score(BASE["hormozi"]["value_equation"]) == 8.0
    assert report.value_score({}) is None
    assert report.value_score({"dream_outcome": 5, "perceived_likelihood": 5,
                               "time_delay": 0, "effort_and_sacrifice": 4}) is None


def test_test_report():
    state = {**BASE, "verdict": {
        "decision": "TEST",
        "reason": "Fintechs already pay lawyers for this.",
        "evidence": ["NDPC fines", "No local tool"],
        "test": {
            "buyer": "Compliance leads", "where_to_find_them": "LinkedIn",
            "offer": "48h audit", "price_ngn": 25000, "people_to_contact": 20,
            "pass_threshold": "3 of 20 pay", "first_message": "Hi,\nquick one", "days": 7,
        },
        "what_would_change_my_mind": "0 of 20 pay",
    }}
    md = report.render_report(state)
    assert md.startswith("# 🧪 TEST: Website scanner")
    assert "₦25,000" in md
    assert "> Hi,\n> quick one" in md
    assert "Stop if" in md
    assert "Cold outreach" in md
    assert "8.0 (strong)" in md
    assert "Buyer is a Lagos fintech" in md


def test_kill_report():
    state = {**BASE, "verdict": {
        "decision": "KILL", "reason": "Needs a CBN licence.", "evidence": ["Lending"],
        "test": None, "what_would_change_my_mind": "A licensed partner",
    }}
    md = report.render_report(state)
    assert md.startswith("# 🛑 KILL")
    assert "day test" not in md
    assert "What would revive it" in md


def test_missing_verdict():
    assert "didn't produce a result" in report.render_report({})
