"""skill-eval scoring engine — rubric-driven, checker-augmented LLM judging.

Loads rubric definitions from JSON config files. Programmatic checkers are
defined per-rubric in separate Python modules. Judge prompts are built from
the rubric config so the same engine works for any skill domain.

Usage:
    from scoring import Rubric

    rubric = Rubric.load("rubrics/consulting.json")
    checks = rubric.run_checkers(output_text)
    judge_prompt = rubric.build_judge_prompt(output_text, checks)
    pairwise_prompt = rubric.build_pairwise_prompt(prompt_text, out_a, out_b, checks_a, checks_b)
"""

import importlib
import json
from pathlib import Path


class Rubric:
    """A scoring rubric loaded from a JSON config file.

    The JSON must contain:
        - version: str
        - name: str
        - tiers: dict[str, str]  (tier number -> tier name)
        - criteria: dict[str, {tier, name, description, anchors}]
        - checker_module: str (dotted Python module path, optional)
        - prompts: dict[str, list[dict]]  (category -> prompt list, optional)
        - approaches: dict[str, dict]  (approach configs, optional)
        - judge_constraints: dict (checker-based score caps, optional)
    """

    def __init__(self, config: dict, base_dir: Path | None = None):
        self.version = config["version"]
        self.name = config["name"]
        self.description = config.get("description", "")
        self.tiers = config["tiers"]  # {"1": "Process Discipline", ...}
        self.criteria = config["criteria"]
        self.prompts = config.get("prompts", {})
        self.approaches = config.get("approaches", {})
        self.judge_constraints = config.get("judge_constraints", {})
        self.base_dir = base_dir or Path(".")

        # Load checker module if specified
        self._checker_mod = None
        checker_path = config.get("checker_module")
        if checker_path:
            try:
                self._checker_mod = importlib.import_module(checker_path)
            except ImportError:
                # Try relative to base_dir
                import sys
                sys.path.insert(0, str(self.base_dir))
                try:
                    self._checker_mod = importlib.import_module(checker_path)
                finally:
                    if str(self.base_dir) in sys.path:
                        sys.path.remove(str(self.base_dir))

    @classmethod
    def load(cls, path: str | Path) -> "Rubric":
        """Load a rubric from a JSON file."""
        path = Path(path)
        with open(path) as f:
            config = json.load(f)
        return cls(config, base_dir=path.parent.parent)

    # ------------------------------------------------------------------
    # Checkers
    # ------------------------------------------------------------------

    def run_checkers(self, text: str) -> dict:
        """Run programmatic checkers on the output text.

        Returns an empty dict if no checker module is configured.
        """
        if self._checker_mod and hasattr(self._checker_mod, "run_checkers"):
            return self._checker_mod.run_checkers(text)
        return {}

    def format_checker_context(self, checks: dict) -> str:
        """Format checker results as a string for the judge prompt."""
        if self._checker_mod and hasattr(self._checker_mod, "format_checker_context"):
            return self._checker_mod.format_checker_context(checks)
        if not checks:
            return ""
        # Generic fallback: key-value listing
        parts = [f"- {k}: {v}" for k, v in checks.items()]
        return "AUTOMATED ANALYSIS:\n" + "\n".join(parts)

    # ------------------------------------------------------------------
    # Judge prompt builders
    # ------------------------------------------------------------------

    def _criteria_text(self) -> str:
        """Build the criteria section for judge prompts."""
        lines = []
        for tier_num in sorted(self.tiers.keys()):
            tier_name = self.tiers[tier_num]
            tier_criteria = {
                k: v for k, v in self.criteria.items()
                if str(v["tier"]) == str(tier_num)
            }
            if not tier_criteria:
                continue
            lines.append(f"\n**Tier {tier_num}: {tier_name}**\n")
            for key, spec in tier_criteria.items():
                lines.append(f"{key} (1-10): {spec['description']}")
                for anchor_score in ["1", "4", "7", "10"]:
                    anchor = spec["anchors"].get(anchor_score, spec["anchors"].get(int(anchor_score), ""))
                    lines.append(f"  {anchor_score} = {anchor}")
        return "\n".join(lines)

    def build_judge_prompt(self, output_text: str, checks: dict) -> str:
        """Build the absolute scoring prompt for LLM judges."""
        checker_ctx = self.format_checker_context(checks)
        n_criteria = len(self.criteria)

        constraint_lines = ""
        if self.judge_constraints and checks:
            parts = []
            for key, max_score in self.judge_constraints.items():
                # Derive the checker field from the constraint key
                # e.g. "hard_gate_max_without_checker" -> check "has_hard_gate"
                if key.endswith("_max_without_checker"):
                    criterion = key.replace("_max_without_checker", "")
                    checker_key = f"has_{criterion}"
                    if checker_key in checks and not checks[checker_key]:
                        parts.append(
                            f"If {checker_key} is No/False, {criterion} cannot score above {max_score}."
                        )
                    # Also handle count-based: evidence_labeling_max_without_checker
                    count_key = f"{criterion}_count"
                    if count_key in checks and checks[count_key] == 0:
                        parts.append(
                            f"If {count_key} is 0, {criterion} cannot score above {max_score}."
                        )
            if parts:
                constraint_lines = "\n" + "\n".join(parts)

        return f"""Score this output on {n_criteria} criteria across {len(self.tiers)} tiers (1-10 each).

{self._criteria_text()}

{checker_ctx}

Use the automated analysis as factual input — e.g., if a checker found 0 instances of a required pattern, the corresponding criterion cannot score high.{constraint_lines}

OUTPUT TO SCORE:
---
{output_text[:8000]}
---

Return a JSON object with all {n_criteria} criteria as integer scores (keys matching the criterion names above) plus a brief "notes" string."""

    def build_pairwise_prompt(
        self,
        prompt_text: str,
        output_a: str,
        output_b: str,
        checks_a: dict,
        checks_b: dict,
    ) -> str:
        """Build prompt for pairwise comparison. Rubric-agnostic."""
        ctx_a = self.format_checker_context(checks_a)
        ctx_b = self.format_checker_context(checks_b)

        tier_descriptions = ", ".join(
            f"{name.lower()} (tier {num})"
            for num, name in sorted(self.tiers.items())
        )

        return f"""You are comparing two outputs for the same prompt. Which is better?

PROMPT: {prompt_text}

=== OUTPUT A ===
{ctx_a}

{output_a[:5000]}

=== OUTPUT B ===
{ctx_b}

{output_b[:5000]}

Consider: {tier_descriptions}.

Which output would you recommend? Reply with a JSON object:
{{"winner": "A" or "B", "confidence": 1-10, "reason": "brief explanation"}}"""

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def score_tier_avg(self, scores: dict, tier: int | str) -> float:
        """Compute average score for a tier."""
        tier_str = str(tier)
        tier_keys = [
            k for k, v in self.criteria.items()
            if str(v["tier"]) == tier_str
        ]
        vals = [scores.get(k, 0) for k in tier_keys]
        valid = [v for v in vals if isinstance(v, (int, float)) and v > 0]
        return round(sum(valid) / len(valid), 1) if valid else 0.0

    def score_summary(self, scores: dict) -> dict:
        """Compute tier averages and overall from raw scores."""
        tier_avgs = {}
        for tier_num, tier_name in self.tiers.items():
            key = f"t{tier_num}_{tier_name.lower().replace(' ', '_')}"
            tier_avgs[key] = self.score_tier_avg(scores, tier_num)

        valid_avgs = [v for v in tier_avgs.values() if v > 0]
        tier_avgs["overall"] = (
            round(sum(valid_avgs) / len(valid_avgs), 1) if valid_avgs else 0.0
        )
        return tier_avgs

    def criteria_keys(self) -> list[str]:
        """Return all criteria keys in tier order."""
        return sorted(self.criteria.keys(), key=lambda k: self.criteria[k]["tier"])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def generate_calibration_template(self, outputs: list[dict]) -> str:
        """Generate a markdown template for human calibration.

        Each item in outputs: {"id": str, "prompt": str, "approach": str, "text": str}
        """
        lines = [
            f"# Human Calibration -- {self.name} Rubric",
            "",
            f"Score each output on the {len(self.criteria)} criteria below (1-10).",
            "After scoring, compare your scores to the automated judges to calibrate.",
            "",
            "## Rubric",
            "",
        ]

        for tier_num in sorted(self.tiers.keys()):
            tier_name = self.tiers[tier_num]
            lines.append(f"### Tier {tier_num}: {tier_name}")
            for key, spec in self.criteria.items():
                if str(spec["tier"]) != str(tier_num):
                    continue
                lines.append(f"\n**{key}** -- {spec['description']}")
                for anchor_score in ["1", "4", "7", "10"]:
                    anchor = spec["anchors"].get(
                        anchor_score,
                        spec["anchors"].get(int(anchor_score), ""),
                    )
                    lines.append(f"- {anchor_score}: {anchor}")
            lines.append("")

        lines.append("---\n")

        for i, item in enumerate(outputs, 1):
            lines.append(f"## Output {i}: {item['id']}")
            lines.append(f"**Prompt:** {item['prompt']}")
            lines.append(f"**Approach:** {item['approach']}")
            lines.append("")
            text = item["text"][:4000]
            if len(item["text"]) > 4000:
                text += "\n\n[... truncated ...]"
            lines.append(f"```\n{text}\n```")
            lines.append("")
            lines.append("### Your scores (1-10):")
            for key in self.criteria:
                lines.append(f"- {key}: ___")
            lines.append("- **Notes:**")
            lines.append("\n---\n")

        lines.append("## Calibration Summary")
        lines.append("")
        tier_headers = " | ".join(
            f"Your T{n} | Judge T{n}" for n in sorted(self.tiers.keys())
        )
        lines.append(f"| Output | {tier_headers} |")
        sep = " | ".join("---" for _ in range(1 + 2 * len(self.tiers)))
        lines.append(f"| {sep} |")
        for i, item in enumerate(outputs, 1):
            cells = " | ".join("___ | ___" for _ in self.tiers)
            lines.append(f"| {item['id']} | {cells} |")
        lines.append("")
        lines.append(
            "**Systematic bias:** Are you consistently higher or lower than the judge?"
        )
        lines.append(
            "**Criterion-level bias:** Which criteria show the largest human-judge delta?"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def get_prompts(self, category: str, prompt_filter: set | None = None) -> list[dict]:
        """Get prompts by category, optionally filtered by ID."""
        if category == "all":
            prompts = []
            for cat_prompts in self.prompts.values():
                prompts.extend(cat_prompts)
        elif "," in (category or ""):
            prompts = []
            for cat in category.split(","):
                prompts.extend(self.prompts.get(cat.strip(), []))
        else:
            prompts = list(self.prompts.get(category or "standard", []))
        if prompt_filter:
            prompts = [p for p in prompts if p["id"] in prompt_filter]
        return prompts
