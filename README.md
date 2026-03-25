# skill-eval -- Rubric-Driven Evaluation for AI Agent Skills

Measure whether your AI skill actually improves output quality. Uses a 3-tier rubric with programmatic checkers, LLM judges, and pairwise comparison with position-swap debiasing.

Built for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills but works with any LLM skill system.

## Why this exists

The old 5-criteria rubric showed the baseline beating the skill on most prompts. The scores were too coarse to detect what the skill actually added. After rebuilding with a 3-tier rubric that separates **process discipline** (what the skill uniquely teaches) from **generic output quality** (what any good LLM does), the picture reversed: the skill adds measurable process discipline that the baseline lacks.

**The lesson:** if your eval rubric doesn't align with what your skill is supposed to improve, you'll measure the wrong thing and conclude the skill doesn't work.

## Methodology

### 3-tier rubric

Each rubric defines criteria across tiers, scored 1-10 with anchored descriptors:

| Tier | What it measures | Example criteria (consulting) |
|------|-----------------|-------------------------------|
| **T1: Process Discipline** | What the skill uniquely adds | Hard gate/assumptions, evidence labeling, framework discipline, devil's advocate |
| **T2: Output Quality** | Generic quality any good LLM achieves | Structure/MECE, evidence depth, actionability, completeness |
| **T3: Decision Quality** | Would you act on this? (anti-circularity guard) | Decision quality |

### Programmatic checkers

Before the LLM judge scores, regex-based checkers run on the raw output to detect objective signals:

- **Evidence labels**: counts `[F]`, `[A]`, `[I]` tags
- **Hard gate**: checks the first 25% of text for assumption/scope statements
- **Devil's advocate**: detects counter-argument / risk sections
- **Framework count**: counts distinct frameworks mentioned

These facts are injected into the judge prompt as constraints -- e.g., "0 evidence labels found, so `evidence_labeling` cannot score above 4." This prevents the LLM judge from hallucinating quality that isn't there.

### Pairwise comparison with position swap

Absolute scores have systematic biases. Pairwise comparison ("which is better?") is more reliable. To eliminate position bias (LLMs tend to prefer the first option), every comparison runs twice with positions swapped. A win requires winning in both directions.

### Human calibration

Generate a template with `calibrate`, score outputs manually, and compare to the automated judge. This reveals systematic biases in the LLM judge that you can correct in the rubric.

## Results (consulting skill, v2 rubric)

15 prompts (easy/medium/hard), baseline vs. skill, Gemini judge:

| Tier | Skill | Baseline | Delta |
|------|-------|----------|-------|
| T1 Process Discipline | 4.8 | 3.2 | **+1.6** |
| T2 Output Quality | 6.1 | 5.5 | +0.6 |
| T3 Decision Quality | 6.4 | 6.0 | +0.4 |
| **Overall** | **5.8** | **4.9** | **+0.9** |

**By difficulty:**
- Easy prompts: T1 gap is +4.8 (skill dominates on process discipline)
- Hard prompts: T1 gap narrows, but skill still wins on pairwise

**Pairwise (position-swap corrected):** 4 skill wins, 4 baseline wins, 7 ties. Skill wins concentrate on hard/ambiguous prompts where process discipline matters most; baseline wins on standard prompts where generic quality suffices.

## Quick start

```bash
# 1. Set your Gemini API key (used for scoring)
export GEMINI_API_KEY=your-key-here

# 2. Run absolute scoring
python3 eval.py absolute \
  --rubric rubrics/consulting.json \
  --skill-dir ~/.claude/skills/management-consulting \
  --model opus

# 3. Run pairwise comparison
python3 eval.py pairwise \
  --rubric rubrics/consulting.json \
  --skill-dir ~/.claude/skills/management-consulting

# 4. Generate human calibration template
python3 eval.py calibrate --rubric rubrics/consulting.json

# 5. View results from a previous run
python3 eval.py report \
  --rubric rubrics/consulting.json \
  --results results/opus/results.jsonl
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--rubric` | Path to rubric JSON file | (required) |
| `--skill-dir` | Path to skill directory containing SKILL.md and references | (required for absolute/pairwise) |
| `--model` | Claude model name for generation | `opus` |
| `--category` | Prompt category: standard, ambiguous, adversarial, all | `all` |
| `--approaches` | Comma-separated approach names | `baseline,full_bundle` |
| `--prompts` | Comma-separated prompt IDs (filter) | all prompts |
| `--pair` | Two approaches to compare in pairwise mode | `baseline,full_bundle` |

### Environment variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Gemini API key for scoring (required) |
| `GEMINI_MODEL` | Gemini model for judging (default: `gemini-2.5-flash`) |
| `CLAUDE_BIN` | Path to Claude CLI binary (default: `claude`) |

## Architecture

```
eval.py                     # CLI: absolute, pairwise, calibrate, report
scoring.py                  # Rubric class: loads JSON, builds prompts, runs checkers
rubrics/
  consulting.json           # Rubric config: criteria, anchors, prompts, approaches
  consulting_checkers.py    # Regex-based programmatic checkers
prompts.json                # Legacy prompt bank (rubric JSON has its own prompts)
references.json             # Legacy approach configs
results/
  opus/                     # Results organized by generation model
    outputs/                # Cached generation outputs
    results.jsonl           # Absolute scoring results
    pairwise.jsonl          # Pairwise comparison results
    human-calibration.md    # Generated calibration template
```

### How scoring works

```
                        +-----------------+
   output text -------->| Programmatic    |----> checker results
                        | Checkers        |      (counts, booleans)
                        +-----------------+
                                |
                                v
                        +-----------------+
   rubric criteria ---->| Judge Prompt    |----> scoring prompt
   checker results ---->| Builder         |      (criteria + anchors + constraints)
                        +-----------------+
                                |
                                v
                        +-----------------+
                        | Gemini API      |----> JSON scores
                        | (JSON schema)   |      {criterion: 1-10, ...}
                        +-----------------+
                                |
                                v
                        +-----------------+
                        | Tier Averages   |----> summary
                        +-----------------+      {t1: X, t2: Y, t3: Z, overall: W}
```

## Creating a custom rubric

1. **Create the rubric JSON** -- define tiers, criteria with anchored descriptors, prompts, and approaches:

```json
{
  "version": "v1.0",
  "name": "my-skill",
  "tiers": {"1": "Core Skill", "2": "Output Quality", "3": "Usefulness"},
  "criteria": {
    "my_criterion": {
      "tier": 1,
      "name": "My Criterion",
      "description": "What this measures",
      "anchors": {
        "1": "Terrible",
        "4": "Below average",
        "7": "Good",
        "10": "Excellent"
      }
    }
  },
  "checker_module": "rubrics.my_checkers",
  "prompts": {
    "standard": [
      {"id": "test-1", "prompt": "Your test prompt", "difficulty": "medium"}
    ]
  },
  "approaches": {
    "baseline": {
      "type": "system_prompt",
      "system_prompt": "You are an expert..."
    },
    "with_skill": {
      "type": "skill_file",
      "file": "SKILL.md"
    }
  }
}
```

2. **Create checkers** (optional) -- implement `run_checkers(text) -> dict` and `format_checker_context(checks) -> str`:

```python
# rubrics/my_checkers.py
import re

def run_checkers(text):
    return {"has_examples": bool(re.search(r"(?i)for example|e\.g\.", text))}

def format_checker_context(checks):
    return f"AUTOMATED ANALYSIS:\n- Has examples: {'Yes' if checks['has_examples'] else 'No'}"
```

3. **Run the eval:**

```bash
python3 eval.py absolute --rubric rubrics/my-skill.json --skill-dir path/to/skill
```

## Requirements

- Python 3.10+
- Gemini API key (`GEMINI_API_KEY` environment variable)
- Claude CLI (`claude`) for generation -- or pre-generate outputs manually
- No pip dependencies -- uses stdlib `urllib` for API calls

## License

MIT
