#!/usr/bin/env python3
"""skill-eval — evaluate AI agent skills with rubric-driven scoring.

3-tier evaluation framework: programmatic checkers + LLM judges + pairwise comparison.
Works with any rubric (JSON config) and any skill directory.

Subcommands:
    absolute   - Score each output independently on all rubric criteria
    pairwise   - Head-to-head comparison with position-swap debiasing
    calibrate  - Generate human calibration template
    report     - Summarize results from a previous run

Usage:
    python3 eval.py absolute --rubric rubrics/consulting.json --skill-dir path/to/skill
    python3 eval.py pairwise --rubric rubrics/consulting.json --skill-dir path/to/skill
    python3 eval.py calibrate --rubric rubrics/consulting.json
    python3 eval.py report --results results/opus/results.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scoring import Rubric

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent

# Gemini API for scoring (Claude CLI unavailable in nested/automated sessions)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"


def _get_gemini_key() -> str:
    """Resolve Gemini API key from environment or .env file."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    env_file = EVAL_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_system_prompt(approach_config: dict, skill_dir: Path) -> str:
    """Build a system prompt from an approach config dict.

    Approach types:
        system_prompt  - use a literal system prompt string
        skill_file     - load a single file from skill_dir
        skill_bundle   - load multiple files from skill_dir
    """
    atype = approach_config["type"]

    if atype == "system_prompt":
        return approach_config["system_prompt"]

    if atype == "skill_file":
        fp = skill_dir / approach_config["file"]
        if not fp.exists():
            print(f"WARNING: skill file not found: {fp}", file=sys.stderr)
            return ""
        return fp.read_text()

    if atype == "skill_bundle":
        parts = []
        for rel_path in approach_config["files"]:
            fp = skill_dir / rel_path
            if fp.exists():
                parts.append(fp.read_text())
            else:
                print(f"WARNING: skill file not found: {fp}", file=sys.stderr)
        return "\n\n---\n\n".join(parts)

    raise ValueError(f"Unknown approach type: {atype}")


def generate_with_claude(system_prompt: str, user_prompt: str,
                         model: str = "opus", timeout: int = 600) -> tuple[str, float]:
    """Generate output using Claude CLI."""
    claude_bin = os.environ.get("CLAUDE_BIN", "claude")
    cmd = [claude_bin, "-p", "--model", model, "--output-format", "text"]
    full_prompt = f"{system_prompt}\n\n---\n\nUser question: {user_prompt}"
    start = time.time()
    try:
        result = subprocess.run(
            cmd, input=full_prompt, capture_output=True, text=True,
            timeout=timeout, env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "cli"},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"ERROR: {e}", time.time() - start
    elapsed = time.time() - start
    if result.returncode != 0:
        return "", elapsed
    return result.stdout.strip(), elapsed


# ---------------------------------------------------------------------------
# Gemini API helpers
# ---------------------------------------------------------------------------

def _gemini_request(url: str, payload: dict, timeout: int = 180) -> dict:
    """Make a Gemini API request using stdlib urllib."""
    import urllib.request
    import urllib.error
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {e.code}: {body[:500]}")


def score_with_gemini(rubric: Rubric, output_text: str, checks: dict) -> dict:
    """Score output using Gemini API with JSON schema enforcement."""
    api_key = _get_gemini_key()
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}

    prompt = rubric.build_judge_prompt(output_text, checks)
    url = f"{GEMINI_URL}/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    schema_props = {k: {"type": "INTEGER"} for k in rubric.criteria}
    schema_props["notes"] = {"type": "STRING"}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": schema_props,
                "required": list(rubric.criteria.keys()),
            },
        },
    }

    try:
        result = _gemini_request(url, payload)
    except Exception as e:
        return {"error": str(e)[:200]}

    candidates = result.get("candidates", [])
    if not candidates:
        return {"error": "no candidates"}

    text = "".join(
        p.get("text", "")
        for p in candidates[0].get("content", {}).get("parts", [])
    )
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return {"error": f"parse: {text[:100]}"}


def pairwise_with_gemini(rubric: Rubric, prompt_text: str,
                         output_a: str, output_b: str,
                         checks_a: dict, checks_b: dict) -> dict:
    """Run a pairwise comparison using Gemini API."""
    api_key = _get_gemini_key()
    if not api_key:
        return {"error": "no key"}

    prompt = rubric.build_pairwise_prompt(prompt_text, output_a, output_b, checks_a, checks_b)
    url = f"{GEMINI_URL}/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "winner": {"type": "STRING", "enum": ["A", "B"]},
                    "confidence": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["winner", "confidence"],
            },
        },
    }

    try:
        result = _gemini_request(url, payload)
    except Exception as e:
        return {"error": str(e)[:200]}

    candidates = result.get("candidates", [])
    if not candidates:
        return {"error": "no candidates"}

    text = "".join(
        p.get("text", "")
        for p in candidates[0].get("content", {}).get("parts", [])
    )
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return {"winner": "A", "confidence": 1, "reason": f"parse error: {text[:50]}"}


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_absolute(args):
    """Run absolute scoring with the rubric's criteria."""
    rubric = Rubric.load(args.rubric)
    skill_dir = Path(args.skill_dir).expanduser()
    prompts = rubric.get_prompts(
        args.category,
        set(args.prompts.split(",")) if args.prompts else None,
    )
    approaches = set(args.approaches.split(","))
    model = args.model

    # Output dirs based on model name
    outputs_dir = EVAL_DIR / "results" / model / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_file = EVAL_DIR / "results" / model / "results.jsonl"

    print(f"Rubric: {rubric.name} {rubric.version} ({len(rubric.criteria)} criteria)")
    print(f"Prompts: {len(prompts)} | Approaches: {approaches} | Model: {model}")
    print(f"Skill dir: {skill_dir}")
    print(f"Judge: Gemini ({GEMINI_MODEL})")
    print()

    results = []
    for prompt in prompts:
        for approach in sorted(approaches):
            tag = f"{prompt['id']}/{approach}"
            output_file = outputs_dir / f"{prompt['id']}_{approach}.txt"

            # Generate or load cached
            if output_file.exists() and output_file.stat().st_size > 100:
                output = output_file.read_text()
                gen_time = 0
                print(f"[{tag}] Cached ({len(output)} chars)")
            else:
                print(f"[{tag}] Generating with {model}...", end=" ", flush=True)
                approach_config = rubric.approaches.get(approach, {})
                if not approach_config:
                    print(f"SKIP (unknown approach)")
                    continue
                system = build_system_prompt(approach_config, skill_dir)
                output, gen_time = generate_with_claude(system, prompt["prompt"], model=model)
                if output and not output.startswith("ERROR:"):
                    output_file.write_text(output)
                    print(f"done ({len(output)} chars, {gen_time:.0f}s)")
                else:
                    print(f"FAILED: {output[:100]}")
                    continue

            # Programmatic checks
            checks = rubric.run_checkers(output)

            # Score with Gemini
            print(f"  Scoring...", end=" ", flush=True)
            scores = score_with_gemini(rubric, output, checks)
            summary = rubric.score_summary(scores) if "error" not in scores else {}

            if summary:
                tier_strs = " ".join(f"{k}={v}" for k, v in summary.items())
                print(tier_strs)
            else:
                print(f"ERROR: {scores.get('error', 'unknown')}")

            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "rubric": rubric.name,
                "rubric_version": rubric.version,
                "generation_model": model,
                "judge_model": GEMINI_MODEL,
                "prompt_id": prompt["id"],
                "difficulty": prompt.get("difficulty", "unknown"),
                "approach": approach,
                "output_len": len(output),
                "gen_time_s": round(gen_time, 1),
                "checks": checks,
                "scores": scores,
                "summary": summary,
            }
            results.append(record)
            with open(results_file, "a") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
            print()

    print_summary(rubric, results)
    print(f"\nResults: {results_file}")


def cmd_pairwise(args):
    """Run pairwise comparison with position-swap debiasing."""
    rubric = Rubric.load(args.rubric)
    prompts = rubric.get_prompts(
        args.category,
        set(args.prompts.split(",")) if args.prompts else None,
    )
    model = args.model
    outputs_dir = EVAL_DIR / "results" / model / "outputs"
    results_file = EVAL_DIR / "results" / model / "pairwise.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Compare baseline vs full_bundle by default, or use --pair
    pair = args.pair.split(",") if args.pair else ["baseline", "full_bundle"]
    if len(pair) != 2:
        print("ERROR: --pair requires exactly 2 approach names", file=sys.stderr)
        sys.exit(1)
    a_name, b_name = pair

    print(f"Pairwise: {len(prompts)} prompts, {a_name} vs {b_name}, 2 directions each")
    print()

    wins = {a_name: 0, b_name: 0, "tie": 0}
    for prompt in prompts:
        a_file = outputs_dir / f"{prompt['id']}_{a_name}.txt"
        b_file = outputs_dir / f"{prompt['id']}_{b_name}.txt"
        if not a_file.exists() or not b_file.exists():
            print(f"[{prompt['id']}] Missing outputs -- run 'absolute' first")
            continue

        a_text = a_file.read_text()
        b_text = b_file.read_text()
        checks_a = rubric.run_checkers(a_text)
        checks_b = rubric.run_checkers(b_text)

        # Direction 1: A=a_name, B=b_name
        print(f"[{prompt['id']}] A={a_name} B={b_name}...", end=" ", flush=True)
        r1 = pairwise_with_gemini(rubric, prompt["prompt"], a_text, b_text, checks_a, checks_b)
        w1 = r1.get("winner", "A")
        print(f"winner={w1} conf={r1.get('confidence', '?')}")

        # Direction 2: swap positions
        print(f"[{prompt['id']}] A={b_name} B={a_name}...", end=" ", flush=True)
        r2 = pairwise_with_gemini(rubric, prompt["prompt"], b_text, a_text, checks_b, checks_a)
        w2 = r2.get("winner", "A")
        print(f"winner={w2} conf={r2.get('confidence', '?')}")

        # Determine overall winner with position-swap correction
        # w1: A wins -> a_name wins; B wins -> b_name wins
        # w2: A wins -> b_name wins; B wins -> a_name wins
        a_votes = (1 if w1 == "A" else 0) + (1 if w2 == "B" else 0)
        b_votes = 2 - a_votes

        if a_votes > b_votes:
            overall = a_name
        elif b_votes > a_votes:
            overall = b_name
        else:
            overall = "tie"
        wins[overall] += 1
        print(f"  => {overall.upper()} ({a_name} {a_votes}/2)")

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "rubric": rubric.name,
            "rubric_version": rubric.version,
            "prompt_id": prompt["id"],
            "difficulty": prompt.get("difficulty", "unknown"),
            "pair": pair,
            "direction1": {"a": a_name, "b": b_name, "winner": w1, "result": r1},
            "direction2": {"a": b_name, "b": a_name, "winner": w2, "result": r2},
            "overall_winner": overall,
            "a_votes": a_votes,
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        print()

    total = sum(wins.values())
    if total > 0:
        print(f"\n{'=' * 50}")
        print("PAIRWISE RESULTS (position-swap corrected)")
        for name in [a_name, b_name, "tie"]:
            count = wins[name]
            pct = 100 * count / total
            print(f"  {name}: {count}/{total} ({pct:.0f}%)")
    print(f"\nResults: {results_file}")


def cmd_calibrate(args):
    """Generate human calibration template."""
    rubric = Rubric.load(args.rubric)
    model = args.model
    outputs_dir = EVAL_DIR / "results" / model / "outputs"

    # Select a representative sample: easy/medium/hard x baseline/skill
    sample_ids = [
        ("market-entry", "baseline"),
        ("market-entry", "full_bundle"),
        ("pe-acquisition", "baseline"),
        ("cost-reduction", "full_bundle"),
        ("ambig-competitive", "baseline"),
        ("ambig-competitive", "full_bundle"),
        ("adv-framework-salad", "baseline"),
        ("adv-framework-salad", "full_bundle"),
    ]

    all_prompts = {}
    for cat_prompts in rubric.prompts.values():
        for p in cat_prompts:
            all_prompts[p["id"]] = p["prompt"]

    outputs = []
    for pid, approach in sample_ids:
        fp = outputs_dir / f"{pid}_{approach}.txt"
        if not fp.exists():
            print(f"Missing: {fp}")
            continue
        outputs.append({
            "id": f"{pid}/{approach}",
            "prompt": all_prompts.get(pid, "?"),
            "approach": approach,
            "text": fp.read_text(),
        })

    if not outputs:
        print("No outputs found. Run 'absolute' first to generate outputs.")
        sys.exit(1)

    template = rubric.generate_calibration_template(outputs)
    out_path = EVAL_DIR / "results" / model / "human-calibration.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template)
    print(f"Calibration template: {out_path} ({len(outputs)} outputs)")
    print("Score each output manually, then compare to judge scores.")


def cmd_report(args):
    """Summarize results from a previous run."""
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    rubric = Rubric.load(args.rubric)
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print("No results found.")
        return

    print_summary(rubric, results)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rubric: Rubric, results: list[dict]):
    """Print a summary table of results."""
    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY -- {rubric.name} {rubric.version}")
    print(f"{'=' * 70}")

    approaches = sorted(set(r["approach"] for r in results))
    for approach in approaches:
        items = [r for r in results if r["approach"] == approach]
        summaries = [r.get("summary", {}) for r in items if r.get("summary")]
        if not summaries:
            continue

        # Average each tier key across all results
        all_keys = set()
        for s in summaries:
            all_keys.update(s.keys())

        avgs = {}
        for key in sorted(all_keys):
            vals = [s[key] for s in summaries if key in s and isinstance(s[key], (int, float))]
            if vals:
                avgs[key] = round(sum(vals) / len(vals), 1)

        parts = " | ".join(f"{k}={v}" for k, v in avgs.items())
        print(f"\n  {approach} (n={len(items)}): {parts}")

    # By difficulty breakdown
    difficulties = sorted(set(r.get("difficulty", "unknown") for r in results))
    if len(difficulties) > 1:
        print(f"\nBy difficulty:")
        for diff in difficulties:
            diff_results = [r for r in results if r.get("difficulty") == diff]
            for approach in approaches:
                items = [r for r in diff_results if r["approach"] == approach]
                summaries = [r.get("summary", {}) for r in items if r.get("summary")]
                if not summaries:
                    continue
                overall = [s.get("overall", 0) for s in summaries]
                avg_overall = round(sum(overall) / len(overall), 1) if overall else 0
                label = approach.upper()
                print(f"  {diff:10s} {label}: overall={avg_overall} (n={len(items)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="skill-eval: evaluate AI agent skills with rubric-driven scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 eval.py absolute --rubric rubrics/consulting.json --skill-dir ~/.claude/skills/management-consulting
  python3 eval.py pairwise --rubric rubrics/consulting.json --skill-dir ~/.claude/skills/management-consulting
  python3 eval.py calibrate --rubric rubrics/consulting.json
  python3 eval.py report --rubric rubrics/consulting.json --results results/opus/results.jsonl
""",
    )

    sub = parser.add_subparsers(dest="command")

    # --- absolute ---
    p_abs = sub.add_parser("absolute", help="Score outputs on all rubric criteria")
    p_abs.add_argument("--rubric", required=True, help="Path to rubric JSON file")
    p_abs.add_argument("--skill-dir", required=True, help="Path to skill directory")
    p_abs.add_argument("--category", default="all", help="Prompt category (default: all)")
    p_abs.add_argument("--prompts", help="Comma-separated prompt IDs to run")
    p_abs.add_argument("--approaches", default="baseline,full_bundle",
                       help="Comma-separated approach names (default: baseline,full_bundle)")
    p_abs.add_argument("--model", default="opus", help="Claude model for generation (default: opus)")

    # --- pairwise ---
    p_pair = sub.add_parser("pairwise", help="Head-to-head comparison with position-swap")
    p_pair.add_argument("--rubric", required=True, help="Path to rubric JSON file")
    p_pair.add_argument("--skill-dir", required=True, help="Path to skill directory")
    p_pair.add_argument("--category", default="all")
    p_pair.add_argument("--prompts", help="Comma-separated prompt IDs")
    p_pair.add_argument("--pair", default="baseline,full_bundle",
                        help="Two approaches to compare (default: baseline,full_bundle)")
    p_pair.add_argument("--model", default="opus")

    # --- calibrate ---
    p_cal = sub.add_parser("calibrate", help="Generate human calibration template")
    p_cal.add_argument("--rubric", required=True, help="Path to rubric JSON file")
    p_cal.add_argument("--model", default="opus")

    # --- report ---
    p_rep = sub.add_parser("report", help="Summarize results from a previous run")
    p_rep.add_argument("--rubric", required=True, help="Path to rubric JSON file")
    p_rep.add_argument("--results", required=True, help="Path to results JSONL file")

    args = parser.parse_args()
    if args.command == "absolute":
        cmd_absolute(args)
    elif args.command == "pairwise":
        cmd_pairwise(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
