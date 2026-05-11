"""Prompt templates for different question types."""

from pathlib import Path

PROMPT_DIR = Path(__file__).parent


def load_prompt_template(question_type: str) -> str:
    """Load prompt template by question type."""
    prompt_file = PROMPT_DIR / f"{question_type}.md"
    if not prompt_file.exists():
        # Fallback to fact_extraction
        prompt_file = PROMPT_DIR / "fact_extraction.md"
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()
