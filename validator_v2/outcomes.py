"""Records every verdict, and what actually happened when you ran the test.

Every run is appended to a JSON-lines log. After running a test, record the real
result from the command line. Recorded results are fed into future verdicts, so
the validator calibrates against your actual tests instead of guessing.

    python -m validator_v2.outcomes list
    python -m validator_v2.outcomes record <run_id> --result pass --paid 3 --contacted 20 \
        --notes "2 paid by transfer, 1 asked for invoice"
    python -m validator_v2.outcomes stats

The log lives at data/validations.jsonl (git-ignored), or VALIDATION_LOG if set.
No ADK imports, so it runs and tests without the SDK.
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "data" / "validations.jsonl"


def log_path() -> Path:
    return Path(os.getenv("VALIDATION_LOG", DEFAULT_LOG))


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or log_path()
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a corrupt line shouldn't break every future run
    return records


def _write_all(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    tmp.replace(path)


def append_record(
    brief: dict[str, Any], verdict: dict[str, Any], path: Path | None = None
) -> str:
    """Log a new run. Returns its run id."""
    path = path or log_path()
    run_id = uuid.uuid4().hex[:6]
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "idea": brief.get("one_liner", ""),
        "buyer": brief.get("buyer", ""),
        "decision": verdict.get("decision"),
        "votes": verdict.get("votes"),
        "confidence": verdict.get("confidence"),
        "reason": verdict.get("reason"),
        "test": verdict.get("test"),
        "outcome": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return run_id


def record_outcome(
    run_id: str,
    result: str,
    paid: int | None = None,
    contacted: int | None = None,
    notes: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Attach the real test result to a run. Raises KeyError if the run isn't found."""
    if result not in ("pass", "fail", "abandoned"):
        raise ValueError("result must be pass, fail or abandoned")
    path = path or log_path()
    records = load_records(path)
    for r in records:
        if r.get("run_id") == run_id:
            r["outcome"] = {
                "result": result,
                "paid": paid,
                "contacted": contacted,
                "notes": notes,
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _write_all(records, path)
            return r
    raise KeyError(run_id)


def summarize_for_prompt(records: list[dict[str, Any]], limit: int = 10) -> str:
    """Recent real results, written for the verdict step to calibrate against."""
    done = [r for r in records if r.get("outcome")][-limit:]
    if not done:
        return "No recorded test results yet. Judge on the evidence alone."
    lines = []
    for r in done:
        o = r["outcome"]
        nums = ""
        if o.get("paid") is not None and o.get("contacted"):
            nums = f" ({o['paid']} of {o['contacted']} paid)"
        price = (r.get("test") or {}).get("price_ngn")
        price_txt = f" at N{price:,}" if isinstance(price, int) else ""
        note = f". Note: {o['notes']}" if o.get("notes") else ""
        lines.append(
            f"- \"{r.get('idea')}\" sold to {r.get('buyer') or 'unknown buyer'}{price_txt}: "
            f"predicted {r.get('decision')} ({r.get('votes')}), "
            f"test {o['result'].upper()}{nums}{note}"
        )
    return "\n".join(lines)


def stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How often TEST verdicts actually passed, overall and by confidence."""
    tests = [r for r in records if r.get("decision") == "TEST"]
    finished = [r for r in tests if r.get("outcome") and r["outcome"]["result"] != "abandoned"]
    by_conf: dict[str, list[int]] = {}
    for r in finished:
        passed = r["outcome"]["result"] == "pass"
        bucket = by_conf.setdefault(r.get("confidence") or "unknown", [0, 0])
        bucket[0] += passed
        bucket[1] += 1
    passed_total = sum(b[0] for b in by_conf.values())
    return {
        "runs": len(records),
        "kills": sum(1 for r in records if r.get("decision") == "KILL"),
        "tests": len(tests),
        "tests_with_results": len(finished),
        "tests_abandoned": sum(
            1 for r in tests if r.get("outcome") and r["outcome"]["result"] == "abandoned"
        ),
        "test_pass_rate": round(passed_total / len(finished), 2) if finished else None,
        "pass_rate_by_confidence": {
            k: f"{v[0]}/{v[1]}" for k, v in sorted(by_conf.items())
        },
    }


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m validator_v2.outcomes")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show logged runs")
    rec = sub.add_parser("record", help="record a test result")
    rec.add_argument("run_id")
    rec.add_argument("--result", required=True, choices=["pass", "fail", "abandoned"])
    rec.add_argument("--paid", type=int)
    rec.add_argument("--contacted", type=int)
    rec.add_argument("--notes", default="")
    sub.add_parser("stats", help="how often TEST verdicts actually passed")
    args = p.parse_args(argv)

    records = load_records()
    if args.cmd == "list":
        if not records:
            print(f"No runs logged yet ({log_path()}).")
        for r in records:
            o = r.get("outcome")
            status = o["result"].upper() if o else "no result yet"
            print(f"{r['run_id']}  {r['created_at'][:10]}  {r['decision']:<4} "
                  f"{r.get('votes') or '':<4} {status:<14} {r.get('idea', '')[:60]}")
    elif args.cmd == "record":
        try:
            r = record_outcome(args.run_id, args.result, args.paid, args.contacted, args.notes)
        except KeyError:
            raise SystemExit(f"No run with id {args.run_id}. Try: list")
        print(f"Recorded {args.result.upper()} for {r['run_id']}: {r.get('idea')}")
    elif args.cmd == "stats":
        print(json.dumps(stats(records), indent=2))


if __name__ == "__main__":
    _main()
