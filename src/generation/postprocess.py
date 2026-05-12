"""Answer post-processing: number verification, format normalization,
empty handling, and length control.
"""

import re


_NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|-?\d+(?:\.\d+)?%?")


def extract_numbers(text: str) -> list[str]:
    """Extract all numbers from text (integers, decimals, percentages)."""
    return _NUMBER_PATTERN.findall(text)


def verify_numbers(answer: str, context_text: str) -> list[str]:
    """Return numbers in answer that do NOT appear in context."""
    ans_nums = set(extract_numbers(answer))
    ctx_nums = set(extract_numbers(context_text))
    return [n for n in ans_nums if n not in ctx_nums]


def normalize_format(text: str) -> str:
    """Standardize percentage and currency formats."""
    # Normalize percentage: 50 % → 50%
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1%", text)
    # Normalize currency: 100 亿元 → 100亿元
    text = re.sub(r"(\d+(?:\.\d+)?)\s*亿", r"\1亿", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*万", r"\1万", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*元", r"\1元", text)
    return text


def handle_empty(answer: str, context_text: str = "") -> str:
    """Return standardized message for empty or uninformative answers."""
    if not answer or not answer.strip():
        return "根据提供的信息无法回答。"
    stripped = answer.strip()
    if stripped in {"无法回答", "不知道", "未提及", "没有相关信息"}:
        return "根据提供的信息无法回答。"
    return answer


def control_length(answer: str, min_len: int = 10, max_len: int = 500) -> str:
    """Enforce answer length constraints."""
    if len(answer) < min_len:
        return answer  # Too short: keep as-is, may trigger re-generation elsewhere
    if len(answer) > max_len:
        # Try to truncate at sentence boundary
        truncated = answer[:max_len]
        last_period = max(truncated.rfind("。"), truncated.rfind("."))
        if last_period > max_len * 0.8:
            return truncated[: last_period + 1]
        return truncated + "..."
    return answer


class PostProcessor:
    """Post-process generated answers for quality and consistency."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 500,
        table_boost: float = 0.1,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.table_boost = table_boost

    def process(
        self, answer: str, context_text: str = "", question_type: str = ""
    ) -> dict:
        """Run full post-processing pipeline.

        Returns dict with:
            - answer: processed answer text
            - hallucinated_numbers: numbers not found in context
            - was_truncated: whether answer was truncated
            - was_empty: whether answer was empty/uninformative
        """
        result = {
            "answer": answer,
            "hallucinated_numbers": [],
            "was_truncated": False,
            "was_empty": False,
        }

        # Empty handling
        processed = handle_empty(answer, context_text)
        if processed != answer:
            result["was_empty"] = True
            result["answer"] = processed
            return result

        # Number verification
        hallucinated = verify_numbers(answer, context_text)
        result["hallucinated_numbers"] = hallucinated

        # Format normalization
        processed = normalize_format(answer)

        # Length control
        controlled = control_length(
            processed, self.min_length, self.max_length
        )
        if len(controlled) < len(processed):
            result["was_truncated"] = True
        result["answer"] = controlled

        return result
