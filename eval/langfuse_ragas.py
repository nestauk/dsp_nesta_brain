from __future__ import annotations

import asyncio
import os

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from config import DEFAULT_EMBEDDINGS_MODEL
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langfuse import Langfuse
from langfuse.client import FetchTracesResponse
from metrics import CorrectedSummarizationScore as SummarizationScore
from ragas import EvaluationDataset
from ragas import MultiTurnSample
from ragas import SingleTurnSample
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.messages import AIMessage
from ragas.messages import HumanMessage
from ragas.metrics import AnswerRelevancy
from ragas.metrics import Faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import answer_relevancy
from ragas.metrics import faithfulness

# from ragas.metrics._simple_criteria import SimpleCriteriaScoreWithoutReference
from ragas.metrics.base import Metric
from ragas.metrics.base import MetricWithEmbeddings
from ragas.metrics.base import MetricWithLLM
from ragas.run_config import RunConfig


if TYPE_CHECKING:
    from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
    from ragas import BaseSample
    from ragas.messages import Message


class SingleTurnSample(SingleTurnSample):
    """Redefining just so it has a pretty_repr method, like MultiTurnSample"""

    def pretty_repr(self) -> str:
        """Print readable input and response"""
        return f"Human: {self.user_input}\n\nAI: {self.response}"


class MultiTurnSample(MultiTurnSample):
    """Redefining just so it has a better pretty_repr method"""

    def pretty_repr(self) -> str:
        """Print readable conversation"""
        format_ = "{actor}: {content}{newlines}"
        return "\n".join(
            [
                format_.format(
                    actor="AI" if type(message) is AIMessage else "Human",
                    content=message.content,
                    newlines="\n\n" if type(message) is AIMessage else "\n",
                )
                for message in self.user_input
            ]
        )


# combining langfuse and ragas

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

langfuse = Langfuse()


def init_ragas_metrics(metrics: List[Metric], llm: LangchainLLMWrapper, embedding: LangchainEmbeddingsWrapper) -> None:
    """Initialise metrics (if necessary) with LLMs or embeddings"""

    # can be useful for defining metrics in the way described here:
    # https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas

    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)


# a way of obtaining scores which may be useful in some circumstances
# adapted from: https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas
async def async_ragas_scores(samples: List[BaseSample], metrics: List[Metric]) -> List[Dict]:
    """Asynchronously derive metric scores for a list of samples"""

    async def sample_scores(sample: BaseSample) -> Dict:
        if isinstance(sample, SingleTurnSample):
            tasks = [asyncio.create_task(metric.single_turn_ascore(sample)) for metric in metrics]
        else:
            tasks = [asyncio.create_task(metric.multi_turn_ascore(sample)) for metric in multi_turn_metrics]
        scores = await asyncio.gather(*tasks)
        return {metrics[i].name: score for i, score in enumerate(scores)}

    multi_turn_metrics = [metric for metric in metrics if hasattr(metric, "_multi_turn_ascore")]
    if not multi_turn_metrics:
        logger.warning("None of the metrics specified have methods for returning scores for multi-turn samples")

    tasks = [asyncio.create_task(sample_scores(sample)) for sample in samples]
    scores = await asyncio.gather(*tasks)

    return scores


def trace_to_conversation(trace: TraceWithDetails, answer_history: List[str]) -> List[Message]:
    """
    Compile the conversation for a MultiTurnSample from its chat history

    TraceWithDetails objects do not seem to distinguish between human and AI messages, which is why
    the answer history is necessary – the AI messages will be in the answer history
    """
    conversation = []
    for message in trace.input["chat_history"]:
        is_ai_message = message["type"] == "ai"  # some of the earlier traces might have mislabelled types.
        #  previous version of this test worked for them: any(message["content"] == output for output in answer_history)
        message_class = AIMessage if is_ai_message else HumanMessage
        message = message_class(content=message["content"])
        conversation.append(message)
    output_answer = trace.output["answer"]
    if type(output_answer) is dict and output_answer.get(
        "quoted_answer"
    ):  # this will be the case if the answer was derived when use_tool_for_citations = True in app.py
        output_answer = output_answer["quoted_answer"]["answer"]
    conversation.append(AIMessage(content=output_answer))
    return conversation


def traces_to_samples(
    traces: Union[FetchTracesResponse, List[TraceWithDetails]],
    filter: Optional[Dict] = None,
    return_dataset: bool = False,
    dataset_path: Optional[str] = None,
) -> List[BaseSample]:
    """Convert a list of traces returned from the Langfuse API into RAGAS samples"""
    # traces are currently deliberately only suitable to be converted into SingleTurnSample objects
    # need to think about MultiTurnSample objects as well

    samples = []
    trace_ids = []  # will need these in order to push scores to langfuse

    if type(traces) is FetchTracesResponse:
        traces = traces.data
    traces = [trace for trace in traces if trace.input["input"]]  # remove empty traces
    if filter:  # filter by user_id, e.g. filter = {'user_id':'helen'} returns only traces with user id helen
        traces = [trace for trace in traces if all(getattr(trace, attr) == value for attr, value in filter.items())]

    traces = sorted(traces, key=lambda trace: trace.timestamp)  # sort so have oldest first – this is necessary for
    # input_appears_in_next_trace_chat_history to work

    answer_history = []
    for i, trace in enumerate(traces):

        if type(trace.output) is dict and "answer" in trace.output:  # if str it will be an error message

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

                answer = trace.output.get("answer")
                if isinstance(answer, dict):
                    answer = answer.get("quoted_answer").get("answer")

                if is_single_turn_sample:
                    sample = SingleTurnSample(
                        user_input=trace.input["input"],
                        retrieved_contexts=[context["page_content"] for context in trace.output["context"]],
                        response=answer,
                    )
                else:
                    sample = MultiTurnSample(
                        user_input=trace_to_conversation(
                            trace, answer_history
                        ),  # contexts are not passed in to MultiTurnSample objects
                    )
                samples.append(sample)
                trace_ids.append(trace.id)
                answer_history = []

    if return_dataset or dataset_path:
        dataset = EvaluationDataset(samples=samples)
        if dataset_path:
            dataset_path.to_csv(dataset_path)

        if return_dataset:
            return dataset

    return samples, trace_ids


def push_scores_to_langfuse(samples: List[BaseSample], trace_ids: List[str], metrics: List[Metric]) -> None:
    """Calculate metric scores for a list of samples and push them to the equivalent traces in Langfuse"""
    if not len(samples) == len(trace_ids):
        raise Exception("trace_ids cannot map precisely onto samples as the two lists are not identical in length")
    ragas_scores = asyncio.run(async_ragas_scores(samples, metrics))
    for i, score_set in enumerate(ragas_scores):
        for score_name, score in score_set.items():
            langfuse.score(trace_id=trace_ids[i], name=score_name, value=score)
    logger.info(f"Pushed scores for {len(samples)} samples to Langfuse")


if __name__ == "__main__":

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=DEFAULT_EMBEDDINGS_MODEL))

    traces = langfuse.fetch_traces()
    samples, trace_ids = traces_to_samples(traces)

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

        context_precision = LLMContextPrecisionWithoutReference()
        metrics = [context_precision, answer_relevancy, faithfulness]

        init_ragas_metrics(
            metrics,
            llm=evaluator_llm,
            embedding=evaluator_embeddings,
        )

        push_scores_to_langfuse(samples, trace_ids, metrics)
