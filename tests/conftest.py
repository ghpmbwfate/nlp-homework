"""Test fixtures - shared across all test modules."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def qianwei_pages():
    """Load 千味央厨 filtered page content for tests."""
    test_dir = Path(__file__).parent
    path = test_dir / "test_pages_qianwei.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def qianwei_questions():
    """Load 千味央厨 test questions."""
    test_dir = Path(__file__).parent
    path = test_dir / "test_questions_qianwei.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
