"""Tests for the majority vote and the outcome log. No ADK or API key needed."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent / "validator_v2"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


consensus = _load("consensus")
outcomes = _load("outcomes")
report = _load("report")

TEST_PLAN = {"buyer": "x", "where_to_find_them": "y", "offer": "z", "price_ngn": 25000,
             "people_to_contact": 20, "pass_threshold": "3 of 20 pay",
             "first_message": "hi", "days": 7}


def v(decision, reason="r"):
    return {"decision": decision, "reason": reason, "evidence": [],
            "test": TEST_PLAN if decision == "TEST" else None,
            "what_would_change_my_mind": "m"}


# --- consensus ---------------------------------------------------------------
def test_unanimous_is_high():
    out = consensus.aggregate_verdicts([v("TEST", "a"), v("TEST", "b"), v("TEST", "c")])
    assert (out["decision"], out["votes"], out["confidence"], out["reason"]) == (
        "TEST", "3/3", "high", "a")


def test_majority_is_low_and_keeps_majority_reasoning():
    out = consensus.aggregate_verdicts([v("TEST", "t1"), v("KILL", "k1"), v("KILL", "k2")])
    assert (out["decision"], out["votes"], out["confidence"], out["reason"]) == (
        "KILL", "2/3", "low", "k1")


def test_tie_goes_to_kill():
    out = consensus.aggregate_verdicts([v("TEST"), v("KILL"), None])
    assert (out["decision"], out["votes"], out["confidence"]) == ("KILL", "1/2", "split")


def test_no_valid_runs():
    assert consensus.aggregate_verdicts([None, {}, {"decision": "MAYBE"}]) is None


def test_input_not_mutated():
    runs = [v("TEST")]
    consensus.aggregate_verdicts(runs)
    assert "votes" not in runs[0]


# --- outcomes ----------------------------------------------------------------
@pytest.fixture
def log(tmp_path):
    return tmp_path / "data" / "validations.jsonl"


def test_full_loop(log):
    brief = {"one_liner": "NDPA audits", "buyer": "Lagos fintechs"}
    verdict = consensus.aggregate_verdicts([v("TEST")] * 3)
    run_id = outcomes.append_record(brief, verdict, log)
    kill_id = outcomes.append_record({"one_liner": "Lending app"},
                                     consensus.aggregate_verdicts([v("KILL")]), log)

    assert outcomes.summarize_for_prompt(outcomes.load_records(log)).startswith(
        "No recorded test results yet")

    outcomes.record_outcome(run_id, "fail", paid=0, contacted=20,
                            notes="liked it, said too pricey", path=log)
    records = outcomes.load_records(log)
    assert [r["run_id"] for r in records] == [run_id, kill_id]

    summary = outcomes.summarize_for_prompt(records)
    assert '"NDPA audits" sold to Lagos fintechs at N25,000' in summary
    assert "predicted TEST (3/3), test FAIL (0 of 20 paid)" in summary
    assert "too pricey" in summary
    assert "Lending app" not in summary  # no result recorded for it

    s = outcomes.stats(records)
    assert s["runs"] == 2 and s["kills"] == 1 and s["tests"] == 1
    assert s["test_pass_rate"] == 0.0
    assert s["pass_rate_by_confidence"] == {"high": "0/1"}


def test_record_unknown_run(log):
    with pytest.raises(KeyError):
        outcomes.record_outcome("nope", "pass", path=log)


def test_bad_result(log):
    with pytest.raises(ValueError):
        outcomes.record_outcome("x", "maybe", path=log)


def test_corrupt_line_is_skipped(log):
    log.parent.mkdir(parents=True)
    log.write_text('{"run_id": "a"}\nnot json\n\n', encoding="utf-8")
    assert [r["run_id"] for r in outcomes.load_records(log)] == ["a"]


def test_missing_log_is_empty(log):
    assert outcomes.load_records(log) == []


# --- report with the new fields ------------------------------------------------
def test_report_shows_vote_debate_and_record_command():
    verdict = consensus.aggregate_verdicts([v("TEST"), v("TEST"), v("KILL")])
    md = report.render_report({
        "idea_brief": {"one_liner": "Idea"},
        "verdict": verdict,
        "run_id": "abc123",
        "bear_case": "- nobody pays",
        "bull_case": "- they pay lawyers",
    })
    assert "Confidence: **low**, only 2/3 runs agreed" in md
    assert "outcomes record abc123 --result pass" in md
    assert "### Case to kill\n\n- nobody pays" in md
    assert "### Case to test\n\n- they pay lawyers" in md
