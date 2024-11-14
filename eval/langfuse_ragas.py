import asyncio
import os

from typing import List

from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langfuse import Langfuse
from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
from metrics import CorrectedSummarizationScore as SummarizationScore
from metrics import summarization_score
from ragas import EvaluationDataset
from ragas import SingleTurnSample
from ragas import evaluate
from ragas.metrics import AnswerRelevancy
from ragas.metrics import Faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import answer_relevancy
from ragas.metrics import faithfulness
from ragas.metrics._simple_criteria import SimpleCriteriaScoreWithoutReference
from ragas_ import async_ragas_scores
from ragas_ import evaluator_embeddings
from ragas_ import evaluator_llm


# combining langfuse and ragas


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

langfuse = Langfuse()


def traces_to_samples(traces: List[TraceWithDetails]) -> List[SingleTurnSample]:
    """Convert a list of traces returned from the Langfuse API into RAGAS samples"""
    # traces are currently deliberately only suitable to be converted into SingleTurnSample objects
    # need to think about MultiTurnSample objects as well

    samples = []

    for trace in traces.data:
        sample = SingleTurnSample(
            user_input=trace.input["input"],
            retrieved_contexts=[context["page_content"] for context in trace.output["context"]],
            response=trace.output["answer"],
        )
        samples.append(sample)

    return samples


if __name__ == "__main__":

    traces = langfuse.fetch_traces()
    samples = traces_to_samples(traces)

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
        metrics = [faithfulness, answer_relevancy, summarization_score, context_precision, simple_criterion]

        init_ragas_metrics(
            metrics,
            llm=evaluator_llm,
            embedding=evaluator_embeddings,
        )

        ragas_scores = asyncio.run(async_ragas_scores(samples, metrics))
        logger.info(ragas_scores)
