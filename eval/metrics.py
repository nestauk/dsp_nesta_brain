from __future__ import annotations

from dataclasses import field
from statistics import mean
from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional
from typing import Set
from typing import cast

import numpy as np

from ragas.embeddings.base import HuggingfaceEmbeddings
from ragas.metrics import RubricsScoreWithoutReference
from ragas.metrics import SummarizationScore
from ragas.metrics.base import MetricType
from ragas.metrics.base import MetricWithEmbeddings
from ragas.metrics.base import SingleTurnMetric


if TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks
    from ragas.dataset_schema import SingleTurnSample


# custom metrics

# This may or may not work – experiment
# The default rubrics can also be used – they are similar to these.
# See https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_domain_specific_rubrics.py
rubrics = {
    "score1_description": "The response is irrelevant and does not answer the question at all.",
    "score2_description": "The response partially answers the question but makes serious omissions, or includes irrelevant information.",  # noqa
    "score3_description": "The response generally answers the question but may lack detail or clarity.",
    "score4_description": "The response answers the question well with only minor issues or missing details.",
    "score5_description": "The response answers the question fully and is clear and detailed.",
}

rubrics_metric_1 = RubricsScoreWithoutReference()  # default rubrics
rubrics_metric_2 = RubricsScoreWithoutReference(rubrics=rubrics)  # bespoke rubrics


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


class ContextSemanticSimilarity(MetricWithEmbeddings, SingleTurnMetric):
    """Retrospectively calculates mean semantic similarity between the input and retrieved contexts"""

    name: str = "input_semantic_similarity"
    _required_columns: Dict[MetricType, Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )
    is_cross_encoder: bool = False
    threshold: Optional[float] = None

    async def _single_turn_ascore(self, sample: SingleTurnSample, *args) -> float:
        """
        Asynchronously retrn metric score for a single turn sample

        Copied from SemanticSimilarity class
        see: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_answer_similarity.py
        """
        row = sample.to_dict()
        return await self._ascore(row, *args)

    async def _ascore(self, row: Dict, *args) -> float:
        """
        Asynchronously return metric score

        Adapted from SemanticSimilarity class
        see: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_answer_similarity.py
        """
        assert self.embeddings is not None, "embeddings must be set"

        user_input = cast(str, row["user_input"])
        retrieved_contexts = cast(str, row["retrieved_contexts"])

        if self.is_cross_encoder and isinstance(self.embeddings, HuggingfaceEmbeddings):
            raise NotImplementedError("async score [ascore()] not implemented for HuggingFace embeddings")
        else:
            embedding_1 = np.array(await self.embeddings.embed_text(user_input))
            context_embeddings = [
                np.array(await self.embeddings.embed_text(context)) for context in retrieved_contexts
            ]
            # Normalization factors of the above embeddings
            norms_1 = np.linalg.norm(embedding_1, keepdims=True)
            context_norms = [np.linalg.norm(embedding, keepdims=True) for embedding in context_embeddings]
            embedding_1_normalized = embedding_1 / norms_1
            context_embeddings_normalized = [
                embedding / context_norms[i] for i, embedding in enumerate(context_embeddings)
            ]
            similarities = [
                (embedding_1_normalized @ context_embedding_normalized.T)
                for context_embedding_normalized in context_embeddings_normalized
            ]
            score = mean(similarities)

        return score
