"""Evaluation dataset schema, loader, and validation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """Structured question for RAG evaluation."""

    id: str
    question: str
    question_ar: str | None = None
    question_en: str | None = None
    ground_truth_articles: list[int] = Field(default_factory=list)
    category: str
    reference_answer: str
    is_refusal_expected: bool = False
    is_ci_subset: bool = False


def get_default_dataset_path() -> Path:
    """Find default path to eval_questions.jsonl."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "data" / "eval" / "eval_questions.jsonl"
        if candidate.exists():
            return candidate
    return Path("data/eval/eval_questions.jsonl")


def load_eval_dataset(
    path: str | Path | None = None,
    subset: str = "all",
) -> list[EvalQuestion]:
    """Load evaluation questions from JSONL file.

    Args:
        path: Path to jsonl file, defaults to data/eval/eval_questions.jsonl.
        subset: 'all' (all items), 'ci' (subset for fast CI runs, <= 20 items),
                or 'full' (all items).

    Returns:
        List of EvalQuestion objects.
    """
    file_path = Path(path) if path else get_default_dataset_path()
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation questions file not found at {file_path}")

    questions: list[EvalQuestion] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            questions.append(EvalQuestion.model_validate(data))

    if subset == "ci":
        ci_questions = [q for q in questions if q.is_ci_subset]
        return ci_questions[:20] if len(ci_questions) >= 20 else ci_questions

    return questions


def validate_eval_dataset(questions: list[EvalQuestion]) -> dict[str, int]:
    """Validate that the dataset meets course standards.

    Checks:
    - >= 50 total questions
    - Arabic and English representation
    - Coverage of key legal categories (contracts, obligations, torts, property, leases, repealed, out_of_scope)
    """
    if len(questions) < 50:
        raise ValueError(f"Dataset has only {len(questions)} questions; minimum required is 50.")

    categories = set(q.category for q in questions)
    required_cats = {
        "contracts",
        "torts",
        "obligations",
        "property",
        "leases",
        "repealed",
        "out_of_scope",
    }
    missing = required_cats - categories
    if missing:
        raise ValueError(f"Dataset missing required legal categories: {missing}")

    twins = [q for q in questions if q.question_ar and q.question_en]
    if len(twins) < 20:
        raise ValueError(f"Found {len(twins)} bilingual twin questions; minimum required is 20.")

    repealed_questions = [q for q in questions if q.category == "repealed"]
    if len(repealed_questions) < 5:
        raise ValueError(
            f"Need at least 5 repealed article questions; found {len(repealed_questions)}"
        )

    refusal_questions = [q for q in questions if q.is_refusal_expected]
    if len(refusal_questions) < 5:
        raise ValueError(
            f"Need at least 5 out-of-scope refusal questions; found {len(refusal_questions)}"
        )

    return {
        "total_questions": len(questions),
        "bilingual_twins": len(twins),
        "repealed_questions": len(repealed_questions),
        "refusal_questions": len(refusal_questions),
        "ci_subset_size": len([q for q in questions if q.is_ci_subset]),
    }
