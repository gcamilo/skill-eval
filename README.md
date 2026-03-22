# skill-eval — Evaluation Framework for AI Agent Skills

A/B test different skill configurations, reference materials, and prompt categories to measure what actually improves AI agent output quality. Built for Claude Code skills but works with any LLM skill system.

## What it does

Instead of guessing whether your skill improvements help, **measure them**:

- **7 reference approaches** (A-G) — test full references, summaries, examples-only, frameworks-only, and more
- **20 prompts across 4 categories** — standard, ambiguous, quick-structure, adversarial
- **Structured JSON scoring** — Gemini API with `response_schema` enforcement for reliable 1-10 scores
- **Pairwise tournament** — "which output is better?" comparisons (more reliable than absolute scores)
- **Automated regression** — weekly cron detects score drops >1 point
- **Human calibration** — template for manual scoring to calibrate the automated judge

## Key finding

In our consulting skill eval (35 runs, 7 approaches), **the skill alone outperformed every reference-augmented variant**:

| Approach | Score |
|---|---|
| Skill only | **7.0/10** |
| + Full MBB case references | 6.8 |
| + Examples only | 6.6 |
| + Summary | 6.2 |
| + Frameworks index | 6.2 |
| + Evidence standards | 6.1 |
| + Everything bundled | **5.0** (worst) |

**More context = worse performance.** The model gets overwhelmed. A well-crafted SKILL.md is the primary lever.

## Quick start

```bash
# Set your Gemini API key
export GEMINI_API_KEY=your-key-here

# Run standard eval with 3 approaches
python eval.py --category standard --approaches A,B,C

# Run with all 7 approaches
python eval.py --category standard --approaches A,B,C,D,E,F,G

# Run pairwise tournament
python eval.py --category standard --approaches A,B,C --mode pairwise

# Run adversarial prompts
python eval.py --category adversarial --approaches A,C

# Run everything
python eval.py --category all --approaches A,B,C,D,E,F,G
```

## Configuration

### Prompts (`prompts.json`)

20 prompts across 4 categories:
- **standard** (5) — clear consulting questions (market entry, margin analysis, PE eval, cost reduction, strategy memo)
- **ambiguous** (5) — vague asks ("we need to grow faster", "competitors are eating our lunch")
- **quick_structure** (5) — exploratory thinking (build vs buy, hire vs promote)
- **adversarial** (5) — anti-pattern requests ("just do a SWOT", "apply all frameworks to Tesla")

### Reference approaches (`references.json`)

7 approaches testing what context helps most:
- **A**: Full reference (skill + MBB case text)
- **B**: Summary (skill + curated 6-point summary)
- **C**: Skill only (baseline)
- **D**: Examples only (skill + 4 worked examples)
- **E**: Frameworks index only (skill + framework lookup table)
- **F**: Evidence standards only (skill + claim labeling rules)
- **G**: Full bundle (skill + all reference files)

### Scoring

Uses Gemini API with `response_mime_type: "application/json"` and `response_schema` for structured integer scores:
- **structure** (1-10): MECE, clear logic flow
- **evidence** (1-10): Claims labeled, numbers sourced
- **actionability** (1-10): Recommendations specific, owned, time-bound
- **framework_fit** (1-10): Right framework for the problem
- **completeness** (1-10): Has bottom line, risks, next actions

### Pairwise mode

For each pair of approaches, the judge picks a winner:
```
"Which consulting output is better? Reply with 'first' or 'second'."
```
Constrained via `response_schema` with enum. More reliable than absolute scoring.

## Adapting for your skill

1. Replace the skill content in `references.json` with your SKILL.md
2. Replace prompts in `prompts.json` with domain-appropriate test cases
3. Adjust scoring criteria in `eval.py` to match your domain
4. Run baseline → modify skill → re-run → compare

## Automated regression (`regression.py`)

Runs the standard eval weekly, compares to previous results, alerts if any criterion drops >1 point. Set up as a cron job:

```bash
# Weekly on Sunday at 2pm UTC
0 14 * * 0 cd /path/to/skill-eval && python regression.py
```

## Human calibration (`human-calibration.md`)

Score 5 outputs manually, compare to the automated judge's scores, adjust the scoring prompt based on disagreements. The template includes rubrics and a comparison table.

## Requirements

- Python 3.10+
- Gemini API key (`GEMINI_API_KEY` env var)
- `pyyaml` (for reading skill files)
- No other dependencies — uses stdlib `urllib` for API calls

## License

MIT
