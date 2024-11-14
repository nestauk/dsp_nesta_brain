import asyncio
import os

from typing import Dict
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import Metric
from ragas.metrics.base import MetricWithEmbeddings
from ragas.metrics.base import MetricWithLLM
from ragas.run_config import RunConfig


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


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
async def async_ragas_scores(samples: List[SingleTurnSample], metrics: List[Metric]) -> List[Dict]:
    """Asynchronously derive metric scores for a list of samples"""

    async def sample_scores(sample: SingleTurnSample) -> Dict:
        tasks = [asyncio.create_task(metric.single_turn_ascore(sample)) for metric in metrics]
        scores = await asyncio.gather(*tasks)
        return {metrics[i].name: score for i, score in enumerate(scores)}

    tasks = [asyncio.create_task(sample_scores(sample)) for sample in samples]
    scores = await asyncio.gather(*tasks)

    return scores
