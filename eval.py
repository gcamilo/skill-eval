#!/usr/bin/env python3
"""Expanded Consulting Skill Eval Harness

Compares 7 reference approaches across 4 prompt categories.
Uses Gemini 3.1 Pro as both generator (via Claude CLI) and judge (via API with JSON schema).

Modes:
  --mode absolute   Score each output independently (default)
  --mode pairwise   Head-to-head comparisons between approaches

Usage:
  python3 eval-expanded.py
  python3 eval-expanded.py --category all --approaches A,B,C,D,E,F,G
  python3 eval-expanded.py --category adversarial --approaches C,G
  python3 eval-expanded.py --mode pairwise --category standard --approaches A,C,G
"""

import argparse
import json
import os
import subprocess
import sys
import time
import itertools
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent
SKILL_DIR = Path.home() / ".claude" / "skills" / "management-consulting"
CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")
RESULTS_FILE = EVAL_DIR / "results-expanded.jsonl"
PAIRWISE_FILE = EVAL_DIR / "results-pairwise.jsonl"

# Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "")
JUDGE_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
JUDGE_URL = "https://generativelanguage.googleapis.com/v1beta"
LOG_DIR = Path("logs")

# ---------------------------------------------------------------------------
# Load config files
# ---------------------------------------------------------------------------

def load_prompts() -> dict:
    with open(EVAL_DIR / "prompts.json") as f:
        return json.load(f)

def load_references() -> dict:
    with open(EVAL_DIR / "references.json") as f:
        return json.load(f)

def load_skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text()

# ---------------------------------------------------------------------------
# Build system prompts for each approach
# ---------------------------------------------------------------------------

def build_system_prompt(approach_key: str, refs: dict, skill_text: str) -> str:
    """Build the system prompt for a given approach key (A-G)."""
    approach = refs["approaches"][approach_key]
    atype = approach["type"]

    if atype == "skill_only":
        return skill_text

    if atype == "inline":
        content_key = approach["content_key"]
        inline = refs["inline_content"][content_key]
        label = "Reference Material (MBB Cases)" if content_key == "full_reference" else "Reference Material (Summary)"
        return f"{skill_text}\n\n## {label}\n{inline}"

    if atype == "skill_files":
        parts = [skill_text]
        for rel_path in approach["files"]:
            full_path = SKILL_DIR / rel_path
            if full_path.exists():
                content = full_path.read_text()
                parts.append(f"\n\n## Reference: {rel_path}\n{content}")
            else:
                print(f"WARNING: File not found: {full_path}", file=sys.stderr)
        return "\n".join(parts)

    raise ValueError(f"Unknown approach type: {atype}")

# ---------------------------------------------------------------------------
# Gemini API helpers
# ---------------------------------------------------------------------------

def _log_gemini(model: str, mode: str, prompt_len: int):
    """Log Gemini API call."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "mode": mode,
        "prompt_chars": prompt_len,
        "caller": "eval-expanded",
    }
    try:
        with open(LOG_DIR / "requests.jsonl", "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _gemini_request(url: str, payload: dict, timeout: int = 120) -> dict:
    """Make a Gemini API request via urllib (no extra deps)."""
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


def gemini_score(output_text: str) -> dict:
    """Score a consulting output using Gemini with JSON schema enforcement."""
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    scoring_prompt = f"""Score this consulting output on 5 criteria (1-10 each).

Criteria:
1. structure: Is it MECE? Clear logic flow? Well-organized?
2. evidence: Are claims labeled (fact/inference/assumption)? Numbers sourced?
3. actionability: Are recommendations specific, owned, time-bound?
4. framework_fit: Did it use the right framework for the problem? Or generic/forced?
5. completeness: Does it have bottom line, risks, next actions?

Consulting output to score:
---
{output_text[:4000]}
---"""

    url = f"{JUDGE_URL}/models/{JUDGE_MODEL}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": scoring_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "structure": {"type": "INTEGER"},
                    "evidence": {"type": "INTEGER"},
                    "actionability": {"type": "INTEGER"},
                    "framework_fit": {"type": "INTEGER"},
                    "completeness": {"type": "INTEGER"},
                    "notes": {"type": "STRING"}
                },
                "required": ["structure", "evidence", "actionability", "framework_fit", "completeness"]
            }
        },
    }

    _log_gemini(JUDGE_MODEL, "score", len(scoring_prompt))
    result = _gemini_request(url, payload)

    candidates = result.get("candidates", [])
    if not candidates:
        return {"structure": 0, "evidence": 0, "actionability": 0, "framework_fit": 0, "completeness": 0, "notes": "no candidates"}

    text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return {"structure": 0, "evidence": 0, "actionability": 0, "framework_fit": 0, "completeness": 0, "notes": f"parse error: {text[:100]}"}


def gemini_pairwise(prompt_text: str, output_a: str, output_b: str) -> str:
    """Ask Gemini which of two outputs is better. Returns 'first' or 'second'."""
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    judge_prompt = f"""You are judging two consulting outputs for the same prompt. Which is better overall (structure, evidence quality, actionability, appropriate framework use, completeness)?

Prompt: {prompt_text}

=== OUTPUT A ===
{output_a[:3000]}

=== OUTPUT B ===
{output_b[:3000]}

Which output is better? Reply with ONLY "first" or "second"."""

    url = f"{JUDGE_URL}/models/{JUDGE_MODEL}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": judge_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 64,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "winner": {"type": "STRING", "enum": ["first", "second"]}
                },
                "required": ["winner"]
            }
        },
    }

    _log_gemini(JUDGE_MODEL, "pairwise", len(judge_prompt))
    result = _gemini_request(url, payload)

    candidates = result.get("candidates", [])
    if not candidates:
        return "first"  # fallback

    text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
    try:
        parsed = json.loads(text.strip())
        return parsed.get("winner", "first")
    except (json.JSONDecodeError, ValueError):
        # Try raw text
        lower = text.strip().lower()
        if "second" in lower:
            return "second"
        return "first"

# ---------------------------------------------------------------------------
# Claude CLI runner
# ---------------------------------------------------------------------------

def run_claude(system_prompt: str, user_prompt: str) -> str:
    """Generate consulting output using Gemini API (Claude CLI times out in this env)."""
    try:
        return call_gemini_text(f"{system_prompt}\n\n---\nUser: {user_prompt}")
    except Exception as e:
        return f"ERROR: {e}"


def call_gemini_text(prompt: str, max_tokens: int = 8192) -> str:
    """Call Gemini API for text generation."""
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
    }).encode()
    url = f"{JUDGE_URL}/models/{JUDGE_MODEL}:generateContent?key={API_KEY}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())
        return result["candidates"][0]["content"]["parts"][0]["text"]

# ---------------------------------------------------------------------------
# Absolute scoring mode
# ---------------------------------------------------------------------------

def run_absolute(prompts_by_cat: dict, categories: list, approach_keys: list,
                 system_prompts: dict):
    """Run absolute scoring: each output scored independently."""
    results = []
    tasks = []
    for cat in categories:
        for p in prompts_by_cat.get(cat, []):
            for ak in approach_keys:
                tasks.append((cat, p, ak))

    total = len(tasks)
    print(f"\n=== ABSOLUTE SCORING: {total} tasks ({len(categories)} categories x {len(approach_keys)} approaches) ===\n")

    for i, (cat, prompt_data, ak) in enumerate(tasks, 1):
        pid = prompt_data["id"]
        approach_name = f"{ak}_{system_prompts[ak]['name']}" if isinstance(system_prompts[ak], dict) else ak
        print(f"[{i}/{total}] {ak} x {pid} ({cat})...", end=" ", flush=True)

        start = time.time()
        output = run_claude(system_prompts[ak]["text"], prompt_data["prompt"])
        elapsed = time.time() - start

        # Rate limit: 1s between Gemini calls
        time.sleep(1.0)
        scores = gemini_score(output)
        numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
        avg = sum(numeric.values()) / max(len(numeric), 1)

        result = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": cat,
            "approach": ak,
            "approach_name": system_prompts[ak]["name"],
            "prompt_id": pid,
            "elapsed_s": round(elapsed, 1),
            "output_len": len(output),
            "scores": scores,
            "avg_score": round(avg, 2),
            "output_preview": output[:200],
            "model": JUDGE_MODEL,
        }
        results.append(result)

        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"avg={avg:.1f} ({elapsed:.1f}s)")

    print_absolute_summary(results, approach_keys, system_prompts)
    return results


def print_absolute_summary(results: list, approach_keys: list, system_prompts: dict):
    """Print summary table for absolute scoring."""
    print("\n" + "=" * 80)
    print("ABSOLUTE SCORING SUMMARY")
    print("=" * 80)

    criteria = ["structure", "evidence", "actionability", "framework_fit", "completeness"]

    # Header
    header = f"{'Approach':<25} {'Avg':>5}"
    for c in criteria:
        header += f" {c[:8]:>8}"
    header += f" {'N':>4}"
    print(header)
    print("-" * len(header))

    for ak in approach_keys:
        ar = [r for r in results if r["approach"] == ak]
        if not ar:
            continue
        avg = sum(r["avg_score"] for r in ar) / len(ar)
        name = system_prompts[ak]["name"]
        row = f"{ak}_{name:<22} {avg:>5.1f}"
        for c in criteria:
            c_avg = sum(r["scores"].get(c, 0) for r in ar) / len(ar)
            row += f" {c_avg:>8.1f}"
        row += f" {len(ar):>4}"
        print(row)

    # Per-category breakdown
    cats = sorted(set(r["category"] for r in results))
    if len(cats) > 1:
        print(f"\n{'--- By Category ---':^80}")
        for cat in cats:
            print(f"\n  {cat}:")
            cat_results = [r for r in results if r["category"] == cat]
            for ak in approach_keys:
                ar = [r for r in cat_results if r["approach"] == ak]
                if not ar:
                    continue
                avg = sum(r["avg_score"] for r in ar) / len(ar)
                name = system_prompts[ak]["name"]
                print(f"    {ak}_{name:<22} avg={avg:.1f}  (n={len(ar)})")

# ---------------------------------------------------------------------------
# Pairwise comparison mode
# ---------------------------------------------------------------------------

def run_pairwise(prompts_by_cat: dict, categories: list, approach_keys: list,
                 system_prompts: dict):
    """Run pairwise comparisons between approaches."""
    if len(approach_keys) < 2:
        print("ERROR: pairwise mode needs at least 2 approaches", file=sys.stderr)
        sys.exit(1)

    # First generate all outputs
    outputs = {}  # (prompt_id, approach) -> output_text
    tasks = []
    for cat in categories:
        for p in prompts_by_cat.get(cat, []):
            for ak in approach_keys:
                tasks.append((cat, p, ak))

    total_gen = len(tasks)
    print(f"\n=== PAIRWISE MODE: Generating {total_gen} outputs ===\n")

    for i, (cat, prompt_data, ak) in enumerate(tasks, 1):
        pid = prompt_data["id"]
        print(f"[{i}/{total_gen}] Generating {ak} x {pid}...", end=" ", flush=True)
        start = time.time()
        output = run_claude(system_prompts[ak]["text"], prompt_data["prompt"])
        elapsed = time.time() - start
        outputs[(pid, ak)] = output
        print(f"done ({elapsed:.1f}s, {len(output)} chars)")

    # Now run all pairwise comparisons
    pairs = list(itertools.combinations(approach_keys, 2))
    prompt_list = []
    for cat in categories:
        for p in prompts_by_cat.get(cat, []):
            prompt_list.append((cat, p))

    total_cmp = len(prompt_list) * len(pairs)
    print(f"\n=== Running {total_cmp} pairwise comparisons ({len(pairs)} pairs x {len(prompt_list)} prompts) ===\n")

    # Win tracking: approach -> {wins, losses}
    wins = {ak: 0 for ak in approach_keys}
    losses = {ak: 0 for ak in approach_keys}
    matchups = {(a, b): {"a_wins": 0, "b_wins": 0} for a, b in pairs}

    cmp_num = 0
    for cat, prompt_data in prompt_list:
        pid = prompt_data["id"]
        for ak_a, ak_b in pairs:
            cmp_num += 1
            out_a = outputs.get((pid, ak_a), "")
            out_b = outputs.get((pid, ak_b), "")

            if not out_a or not out_b:
                print(f"[{cmp_num}/{total_cmp}] SKIP {ak_a} vs {ak_b} x {pid} (missing output)")
                continue

            print(f"[{cmp_num}/{total_cmp}] {ak_a} vs {ak_b} x {pid}...", end=" ", flush=True)
            time.sleep(1.0)  # rate limit
            winner = gemini_pairwise(prompt_data["prompt"], out_a, out_b)

            if winner == "first":
                wins[ak_a] += 1
                losses[ak_b] += 1
                matchups[(ak_a, ak_b)]["a_wins"] += 1
                winner_label = ak_a
            else:
                wins[ak_b] += 1
                losses[ak_a] += 1
                matchups[(ak_a, ak_b)]["b_wins"] += 1
                winner_label = ak_b

            print(f"winner={winner_label}")

            # Log result
            pw_result = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "category": cat,
                "prompt_id": pid,
                "approach_a": ak_a,
                "approach_b": ak_b,
                "winner": winner,
                "winner_approach": winner_label,
                "model": JUDGE_MODEL,
            }
            with open(PAIRWISE_FILE, "a") as f:
                f.write(json.dumps(pw_result) + "\n")

    # Print standings
    print("\n" + "=" * 60)
    print("PAIRWISE TOURNAMENT STANDINGS")
    print("=" * 60)

    standings = []
    for ak in approach_keys:
        total = wins[ak] + losses[ak]
        win_rate = wins[ak] / total * 100 if total > 0 else 0
        standings.append((ak, system_prompts[ak]["name"], wins[ak], losses[ak], total, win_rate))

    standings.sort(key=lambda x: x[5], reverse=True)

    print(f"{'Rank':<5} {'Approach':<25} {'W':>4} {'L':>4} {'Total':>6} {'Win%':>6}")
    print("-" * 50)
    for rank, (ak, name, w, l, t, wr) in enumerate(standings, 1):
        print(f"  {rank:<3} {ak}_{name:<22} {w:>4} {l:>4} {t:>6} {wr:>5.1f}%")

    # Head-to-head matrix
    print(f"\n{'--- Head-to-Head ---'}")
    for (ak_a, ak_b), m in matchups.items():
        total = m["a_wins"] + m["b_wins"]
        if total == 0:
            continue
        print(f"  {ak_a} vs {ak_b}: {m['a_wins']}-{m['b_wins']}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Expanded consulting skill eval")
    parser.add_argument("--mode", choices=["absolute", "pairwise"], default="absolute",
                        help="Scoring mode (default: absolute)")
    parser.add_argument("--category", default="standard",
                        help="Prompt category: standard|ambiguous|quick_structure|adversarial|all (default: standard)")
    parser.add_argument("--approaches", default="A,B,C",
                        help="Comma-separated approach keys: A,B,C,D,E,F,G (default: A,B,C)")
    args = parser.parse_args()

    # Validate env
    if not API_KEY:
        # Try loading from .env
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            globals()["API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Export it or add to .env", file=sys.stderr)
        sys.exit(1)

    # Parse categories
    prompts_data = load_prompts()
    all_cats = list(prompts_data.keys())
    if args.category == "all":
        categories = all_cats
    else:
        categories = [c.strip() for c in args.category.split(",")]
        for c in categories:
            if c not in prompts_data:
                print(f"ERROR: Unknown category '{c}'. Available: {all_cats}", file=sys.stderr)
                sys.exit(1)

    # Parse approaches
    refs = load_references()
    approach_keys = [a.strip().upper() for a in args.approaches.split(",")]
    valid_keys = list(refs["approaches"].keys())
    for ak in approach_keys:
        if ak not in valid_keys:
            print(f"ERROR: Unknown approach '{ak}'. Available: {valid_keys}", file=sys.stderr)
            sys.exit(1)

    # Build system prompts
    skill_text = load_skill_text()
    system_prompts = {}
    for ak in approach_keys:
        text = build_system_prompt(ak, refs, skill_text)
        system_prompts[ak] = {
            "name": refs["approaches"][ak]["name"],
            "text": text,
        }

    print(f"Config: mode={args.mode}, categories={categories}, approaches={approach_keys}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Skill dir: {SKILL_DIR}")

    if args.mode == "absolute":
        run_absolute(prompts_data, categories, approach_keys, system_prompts)
    elif args.mode == "pairwise":
        run_pairwise(prompts_data, categories, approach_keys, system_prompts)


if __name__ == "__main__":
    main()
