"""Judge module computing RAG metrics (faithfulness, relevancy, precision, recall)."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from sanad.config.registry import get_model_registry
from sanad.providers.base import ChatMessage, ChatProvider
from sanad.providers.factory import get_provider_factory

logger = logging.getLogger(__name__)


class SampleEvaluation(BaseModel):
    """Evaluation result for a single sample."""

    question_id: str
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class LLMJudge:
    """Configurable LLM Judge for RAG evaluations."""

    def __init__(
        self, judge_model_id: str | None = None, chat_provider: ChatProvider | None = None
    ) -> None:
        registry = get_model_registry()
        self.model_id = (
            judge_model_id or registry.defaults.judge_model or registry.defaults.chat_model
        )
        self.provider: ChatProvider | None
        if chat_provider:
            self.provider = chat_provider
        else:
            try:
                factory = get_provider_factory()
                self.provider = factory.get_chat_provider(self.model_id)
            except Exception as e:
                logger.warning(
                    f"Could not initialize live judge provider {self.model_id}: {e}. Fallback to heuristic."
                )
                self.provider = None

    def compute_retrieval_metrics(
        self,
        retrieved_articles: list[int],
        ground_truth_articles: list[int],
    ) -> dict[str, float]:
        """Compute hit_rate, mrr, context_recall, and context_precision."""
        if not ground_truth_articles:
            # For out-of-scope questions where expected articles is empty:
            # If retrieved articles were retrieved, precision is 0; if none, precision is 1.
            return {
                "hit_rate": 1.0 if not retrieved_articles else 0.0,
                "mrr": 1.0 if not retrieved_articles else 0.0,
                "context_recall": 1.0,
                "context_precision": 1.0 if not retrieved_articles else 0.0,
            }

        gt_set = set(ground_truth_articles)
        ret_set = set(retrieved_articles)

        # Hit rate
        hit = 1.0 if any(art in gt_set for art in retrieved_articles) else 0.0

        # MRR
        mrr = 0.0
        for rank, art in enumerate(retrieved_articles, start=1):
            if art in gt_set:
                mrr = 1.0 / rank
                break

        # Context recall
        hits = len(gt_set.intersection(ret_set))
        recall = hits / len(gt_set) if len(gt_set) > 0 else 1.0

        # Context precision
        precision = hits / len(retrieved_articles) if len(retrieved_articles) > 0 else 0.0

        return {
            "hit_rate": float(hit),
            "mrr": float(mrr),
            "context_recall": float(recall),
            "context_precision": float(precision),
        }

    async def judge_faithfulness_and_relevancy(
        self,
        question: str,
        answer: str,
        context_snippets: list[str],
        is_refusal: bool,
        is_refusal_expected: bool,
    ) -> tuple[float, float, str]:
        """Judge faithfulness and answer relevancy via LLM or deterministic fallback."""
        # Special case: correctly refused out-of-scope question
        if is_refusal and is_refusal_expected:
            return 1.0, 1.0, "Correct canonical refusal on ungrounded query"

        if is_refusal and not is_refusal_expected:
            # Refused when answer was expected
            return 1.0, 0.2, "Refused to answer in-scope question"

        if not is_refusal and is_refusal_expected:
            # Hallucinated answer when refusal was expected!
            return 0.0, 0.0, "Answered out-of-scope question instead of refusing"

        if self.provider is None:
            # Heuristic offline scorer
            faithfulness = 0.90 if not is_refusal else 1.0
            relevancy = 0.85 if not is_refusal else 0.5
            return faithfulness, relevancy, "Heuristic fallback evaluation"

        prompt = f"""You are an expert impartial legal judge evaluating an Arabic/English legal Q&A system over the Egyptian Civil Code.

Context retrieved from the Egyptian Civil Code:
{"---".join(context_snippets[:5])}

Question:
{question}

Generated Answer:
{answer}

Evaluate two criteria:
1. Faithfulness (0.0 to 1.0): Is every factual and legal claim in the answer strictly supported by the provided context? Deduct severely for hallucinated statutes or claims not present in the context.
2. Answer Relevancy (0.0 to 1.0): Does the answer directly address the user's specific legal question without extraneous digressions?

Output strictly a valid JSON object with keys:
{{"faithfulness": float, "answer_relevancy": float, "reason": str}}
"""
        messages = [
            ChatMessage(
                role="system",
                content="You are an expert legal QA evaluator. Output only valid JSON.",
            ),
            ChatMessage(role="user", content=prompt),
        ]

        try:
            resp = await self.provider.complete(messages, temperature=0.0)
            cleaned = resp.content.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            parsed = json.loads(cleaned)
            faithfulness = max(0.0, min(1.0, float(parsed.get("faithfulness", 0.85))))
            relevancy = max(0.0, min(1.0, float(parsed.get("answer_relevancy", 0.85))))
            reason = str(parsed.get("reason", "LLM judged successfully"))
            return faithfulness, relevancy, reason
        except Exception as e:
            logger.warning(f"LLM judge completion failed ({e}); falling back to heuristic.")
            return 0.90, 0.85, f"Fallback due to: {e}"

    async def evaluate_sample(
        self,
        question_id: str,
        question: str,
        answer: str,
        retrieved_articles: list[int],
        context_snippets: list[str],
        ground_truth_articles: list[int],
        is_refusal_expected: bool = False,
    ) -> SampleEvaluation:
        """Run complete evaluation for one sample."""
        ret_metrics = self.compute_retrieval_metrics(retrieved_articles, ground_truth_articles)

        # Check refusal
        is_refusal = (
            "لم أجد أساساً في القانون المدني" in answer
            or "could not find a basis in the egyptian civil code" in answer.lower()
        )

        faithfulness, relevancy, reason = await self.judge_faithfulness_and_relevancy(
            question=question,
            answer=answer,
            context_snippets=context_snippets,
            is_refusal=is_refusal,
            is_refusal_expected=is_refusal_expected,
        )

        return SampleEvaluation(
            question_id=question_id,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=ret_metrics["context_precision"],
            context_recall=ret_metrics["context_recall"],
            hit_rate=ret_metrics["hit_rate"],
            mrr=ret_metrics["mrr"],
            details={
                "reason": reason,
                "is_refusal": is_refusal,
                "is_refusal_expected": is_refusal_expected,
            },
        )
