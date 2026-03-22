#!/usr/bin/env python3
"""Weekly regression: run standard eval, compare to previous, alert on drops.

Runs eval-expanded.py --category standard --approaches A,C
Compares to last results in data/consulting-eval/history/
If any criterion drops >1 point, alerts to Discord.
Saves timestamped results to data/consulting-eval/history/
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path.home() / "orchestrator" / "data" / "consulting-eval"
HISTORY_DIR = EVAL_DIR / "history"
EVAL_SCRIPT = EVAL_DIR / "eval-expanded.py"
DISCORD_SCRIPT = Path.home() / "orchestrator" / "scripts" / "discord"
VENV_PYTHON = Path.home() / "orchestrator" / "venv" / "bin" / "python3"
RESULTS_FILE = EVAL_DIR / "results-expanded.jsonl"

APPROACHES = "A,C"
CATEGORY = "standard"
CRITERIA = ["structure", "evidence", "actionability", "framework_fit", "completeness"]
DROP_THRESHOLD = 1.0  # alert if any criterion drops more than this


def get_python():
    """Find the right Python interpreter."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def get_latest_history() -> Path | None:
    """Find the most recent history file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(HISTORY_DIR.glob("*.jsonl"), reverse=True)
    return files[0] if files else None


def load_results(path: Path) -> list[dict]:
    """Load JSONL results file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_averages(results: list[dict]) -> dict:
    """Compute per-approach, per-criterion averages."""
    avgs = {}
    for r in results:
        ak = r.get("approach", r.get("approach_name", "?"))
        if ak not in avgs:
            avgs[ak] = {c: [] for c in CRITERIA}
            avgs[ak]["avg_score"] = []
        for c in CRITERIA:
            val = r.get("scores", {}).get(c, 0)
            avgs[ak][c].append(val)
        avgs[ak]["avg_score"].append(r.get("avg_score", 0))

    # Convert lists to means
    for ak in avgs:
        for key in avgs[ak]:
            vals = avgs[ak][key]
            avgs[ak][key] = sum(vals) / len(vals) if vals else 0
    return avgs


def discord_alert(message: str):
    """Send alert to Discord #orchestrator channel."""
    try:
        subprocess.run(
            [str(DISCORD_SCRIPT), "reply", "orchestrator", message],
            timeout=30, capture_output=True,
        )
    except Exception as e:
        print(f"Discord alert failed: {e}", file=sys.stderr)


def main():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previous results file to get clean run
    if RESULTS_FILE.exists():
        RESULTS_FILE.rename(RESULTS_FILE.with_suffix(".jsonl.bak"))

    # Run the eval
    python = get_python()
    print(f"Running eval: {python} {EVAL_SCRIPT} --category {CATEGORY} --approaches {APPROACHES}")
    result = subprocess.run(
        [python, str(EVAL_SCRIPT), "--category", CATEGORY, "--approaches", APPROACHES],
        timeout=1800,  # 30 min max
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        msg = f"Consulting eval regression FAILED (exit {result.returncode}): {result.stderr[:300]}"
        print(msg, file=sys.stderr)
        discord_alert(msg)
        sys.exit(1)

    print(result.stdout)

    # Load new results
    if not RESULTS_FILE.exists():
        print("ERROR: No results file produced", file=sys.stderr)
        sys.exit(1)

    new_results = load_results(RESULTS_FILE)
    if not new_results:
        print("ERROR: Empty results file", file=sys.stderr)
        sys.exit(1)

    # Save to history
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    history_file = HISTORY_DIR / f"eval-{ts}.jsonl"
    with open(history_file, "w") as f:
        for r in new_results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {history_file}")

    # Compare to previous
    new_avgs = compute_averages(new_results)
    prev_file = get_latest_history()
    # The file we just wrote is the latest; find the one before it
    history_files = sorted(HISTORY_DIR.glob("*.jsonl"), reverse=True)
    prev_file = history_files[1] if len(history_files) > 1 else None

    if prev_file is None:
        print("No previous results to compare against. First run baseline stored.")
        discord_alert(
            f"**Consulting Eval Baseline**\n"
            f"First regression run complete. {len(new_results)} results saved.\n"
            f"Approaches: {APPROACHES} | Category: {CATEGORY}"
        )
        return

    prev_results = load_results(prev_file)
    prev_avgs = compute_averages(prev_results)

    # Compare and detect drops
    drops = []
    for ak in new_avgs:
        if ak not in prev_avgs:
            continue
        for c in CRITERIA:
            new_val = new_avgs[ak].get(c, 0)
            old_val = prev_avgs[ak].get(c, 0)
            delta = new_val - old_val
            if delta < -DROP_THRESHOLD:
                drops.append({
                    "approach": ak,
                    "criterion": c,
                    "old": round(old_val, 1),
                    "new": round(new_val, 1),
                    "delta": round(delta, 1),
                })

    # Build report
    report_lines = [f"**Consulting Eval Regression** ({ts})"]
    report_lines.append(f"Approaches: {APPROACHES} | Category: {CATEGORY} | N={len(new_results)}")
    report_lines.append("")

    for ak in sorted(new_avgs.keys()):
        avg = new_avgs[ak].get("avg_score", 0)
        prev_avg = prev_avgs.get(ak, {}).get("avg_score", 0)
        delta = avg - prev_avg
        arrow = "+" if delta >= 0 else ""
        report_lines.append(f"**{ak}**: avg={avg:.1f} ({arrow}{delta:.1f})")

    if drops:
        report_lines.append("")
        report_lines.append("**REGRESSIONS DETECTED:**")
        for d in drops:
            report_lines.append(
                f"  - {d['approach']}/{d['criterion']}: {d['old']} -> {d['new']} ({d['delta']:+.1f})"
            )

    report = "\n".join(report_lines)
    print(report)

    if drops:
        discord_alert(report)
        print(f"\nALERT: {len(drops)} regression(s) detected and reported to Discord")
    else:
        print("\nNo regressions detected. All criteria within threshold.")


if __name__ == "__main__":
    main()
