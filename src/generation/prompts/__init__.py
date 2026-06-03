"""Prompt templates for different question types."""

from pathlib import Path

PROMPT_DIR = Path(__file__).parent


def load_prompt_template(question_type: str, version: str = "v1") -> str:
    """Load prompt template by question type and version.

    Args:
        question_type: one of fact_extraction / analysis_summary /
            chart_understanding / comparison.
        version: prompt version. "v1" loads `{type}.md`; any other value
            loads `{type}_{version}.md` and falls back to v1 if missing.
    """
    if version == "v1":
        prompt_file = PROMPT_DIR / f"{question_type}.md"
    else:
        prompt_file = PROMPT_DIR / f"{question_type}_{version}.md"
        if not prompt_file.exists():
            prompt_file = PROMPT_DIR / f"{question_type}.md"
    if not prompt_file.exists():
        # Final fallback to fact_extraction v1
        prompt_file = PROMPT_DIR / "fact_extraction.md"
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()
