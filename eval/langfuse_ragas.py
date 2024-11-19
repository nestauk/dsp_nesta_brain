from __future__ import annotations

import asyncio
import os

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional

from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langfuse import Langfuse
from langfuse.client import FetchTracesResponse
from metrics import ContextSemanticSimilarity
from metrics import CorrectedSummarizationScore as SummarizationScore
from metrics import summarization_score
from ragas import EvaluationDataset
from ragas import MultiTurnSample
from ragas import SingleTurnSample
from ragas import evaluate
from ragas.messages import AIMessage
from ragas.messages import HumanMessage
from ragas.metrics import AnswerRelevancy
from ragas.metrics import Faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import answer_relevancy
from ragas.metrics import faithfulness
from ragas.metrics._simple_criteria import SimpleCriteriaScoreWithoutReference
from ragas_ import async_ragas_scores
from ragas_ import evaluator_embeddings
from ragas_ import evaluator_llm


if TYPE_CHECKING:
    from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
    from ragas import BaseSample
    from ragas.messages import Message

# combining langfuse and ragas


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

langfuse = Langfuse()


def trace_to_conversation(trace: TraceWithDetails, answer_history: List[str]) -> List[Message]:
    """
    Compile the conversation for a MultiTurnSample from its chat history

    TraceWithDetails objects do not seem to distinguish between human and AI messages, which is why
    the answer history is necessary – the AI messages will be in the answer history
    """
    conversation = []
    for message in trace.input["chat_history"]:
        is_ai_message = any(message["content"] == output for output in answer_history)
        message_class = AIMessage if is_ai_message else HumanMessage
        message = message_class(content=message["content"])
        conversation.append(message)
    return conversation


def traces_to_samples(
    traces: FetchTracesResponse,
    filter: Optional[Dict] = None,
    return_dataset: bool = False,
    dataset_path: Optional[str] = None,
) -> List[BaseSample]:
    """Convert a list of traces returned from the Langfuse API into RAGAS samples"""
    # traces are currently deliberately only suitable to be converted into SingleTurnSample objects
    # need to think about MultiTurnSample objects as well

    samples = []

    traces = traces.data
    if filter:  # filter by user_id, e.g. filter = {'user_id':'helen'} returns only traces with user id helen
        traces = [trace for trace in traces if all(getattr(trace, attr) == value for attr, value in filter.items())]

    traces = sorted(traces, key=lambda trace: trace.timestamp)  # sort so have oldest first – this is necessary for
    # input_appears_in_next_trace_chat_history to work

    answer_history = []
    for i, trace in enumerate(traces):

        is_single_turn_sample = False

        next_trace = traces[i + 1] if i < len(traces) - 1 else None
        if next_trace:
            input_appears_in_next_trace_chat_history = any(
                message["content"] == trace.input["input"] for message in next_trace.input["chat_history"]
            )
        else:
            input_appears_in_next_trace_chat_history = False

        if input_appears_in_next_trace_chat_history:
            # if input_appears_in_next_trace_chat_history = True, the message will eventually appear in
            # the conversation of a MultiTurnSample, so do not instantiate a sample for it
            # but do retain the answer history for when the MultiTurnSample is eventually instantiated
            answer_history.append(trace.output["answer"])
        else:
            # trace is either a SingleTurnSample or the last trace defining a MultiTurnSample
            is_single_turn_sample = len(trace.input["chat_history"]) == 1

            if is_single_turn_sample:
                sample = SingleTurnSample(
                    user_input=trace.input["input"],
                    retrieved_contexts=[context["page_content"] for context in trace.output["context"]],
                    response=trace.output["answer"],
                )
            else:
                sample = MultiTurnSample(
                    user_input=trace_to_conversation(
                        trace, answer_history
                    ),  # contexts are not passed in to MultiTurnSample objects
                )
            samples.append(sample)
            answer_history = []

    if return_dataset or dataset_path:
        dataset = EvaluationDataset(samples=samples)
        if dataset_path:
            dataset_path.to_csv(dataset_path)

        if return_dataset:
            return dataset

    return samples


if __name__ == "__main__":

    traces = langfuse.fetch_traces()
    samples = traces_to_samples(traces)  # , filter={"user_id": "helen"})

    if False:
        # one way of defining which metrics to use and getting evaluation scores
        metrics = [
            AnswerRelevancy(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm),
            SummarizationScore(llm=evaluator_llm),
        ]

        dataset = EvaluationDataset(samples=samples)
        results = evaluate(dataset=dataset, metrics=metrics)
        df = results.to_pandas()
        logger.info(df)

    else:
        # another way
        from ragas_ import init_ragas_metrics

        context_precision = LLMContextPrecisionWithoutReference()
        simple_criterion = SimpleCriteriaScoreWithoutReference(
            name="my_test",
            definition="Score responses in range of 0 to 5 based on factors such as grammar, relevance, and coherence.",
        )  # trivial example for experimentation
        metrics = [
            faithfulness,
            answer_relevancy,
            summarization_score,
            context_precision,
            simple_criterion,
            ContextSemanticSimilarity(),
        ]

        init_ragas_metrics(
            metrics,
            llm=evaluator_llm,
            embedding=evaluator_embeddings,
        )

        ragas_scores = asyncio.run(async_ragas_scores(samples, metrics))
        logger.info(ragas_scores)
