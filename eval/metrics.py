from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict

from ragas.metrics import SummarizationScore
from ragas.metrics.base import MetricType


if TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks


# custom metrics


class CorrectedSummarizationScore(SummarizationScore):
    """Corrected SummarizationScore

    I think there is a typo in the _ascore method of SummarizationScore
    see: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_summarization.py
    This class exists just to correct the typo
    The documentation mentions that the metric works on retrieved contexts:
    https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/summarization_score/
    I have found an error is thrown in the _ascore method without this correction
    """

    def __init__(self, *args, **kwargs) -> None:
        required_columns = {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",
                "response",
            }
        }
        super().__init__(*args, _required_columns=required_columns, **kwargs)

    async def _ascore(self, row: Dict, callbacks: Callbacks) -> float:

        text: str = "\n".join(
            row["retrieved_contexts"]
        )  # typo corrected here – should be retrieved_contexts, not reference_contexts
        summary: str = row["response"]
        keyphrases = await self._extract_keyphrases(text, callbacks)
        questions = await self._get_questions(text, keyphrases, callbacks)
        answers = await self._get_answers(questions, summary, callbacks)

        scores = {}
        qa_score = self._compute_qa_score(answers)
        scores["qa_score"] = qa_score
        if self.length_penalty:
            conciseness_score = self._compute_conciseness_score(text, summary)
            scores["conciseness_score"] = conciseness_score
        return self._compute_score(scores)


summarization_score = CorrectedSummarizationScore()
