"""Combines several independent verdict runs into one decision.

LLMs lean toward encouragement, so a single run says TEST too often. Running the
verdict several times and taking the majority shows how stable the decision is:
a 3/3 decision is solid; 2/3 means it's a judgement call.

No ADK imports, so this is unit-testable.
"""

from collections import Counter
from typing import Any


def aggregate_verdicts(runs: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Return the majority verdict, with `votes` and `confidence` added.

    A tie goes to KILL: when the runs can't agree, the idea hasn't earned a test.
    """
    valid = [r for r in runs if r and r.get("decision") in ("KILL", "TEST")]
    if not valid:
        return None

    counts = Counter(r["decision"] for r in valid)
    kills, tests = counts.get("KILL", 0), counts.get("TEST", 0)
    decision = "TEST" if tests > kills else "KILL"
    agreeing = max(kills, tests)

    chosen = dict(next(r for r in valid if r["decision"] == decision))
    chosen["votes"] = f"{agreeing}/{len(valid)}"
    if agreeing == len(valid):
        chosen["confidence"] = "high"
    elif kills == tests:
        chosen["confidence"] = "split"
    else:
        chosen["confidence"] = "low"
    return chosen
