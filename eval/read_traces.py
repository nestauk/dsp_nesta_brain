from __future__ import annotations

import os
import re
import sys

from datetime import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import List

from dotenv import load_dotenv
from dsp_nesta_brain import logger
from eval.langfuse_ragas import traces_to_samples
from langfuse import Langfuse
from utils import bold
from utils import yesno


if TYPE_CHECKING:
    from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails


LANGFUSE_TRACE_LIMIT = 100


def is_admin(trace: TraceWithDetails) -> bool:
    """Test whether trace was by an admin rather than another user"""
    common_test_questions = [
        "I am designing a citizen engagement project with a local authority. What considerations should I bear in mind when putting together the proposal?",  # noqa
        "Tell me more about communication strategy",
        "What work has Nesta done on heat pumps?",
        "Who has data science skills at Nesta?",
    ]
    if trace.user_id == "helen" or "admin_test" in trace.tags:
        return True
    return trace.input["input"].strip() in common_test_questions


def collate_traces(ultimate_filters: Dict) -> List[TraceWithDetails]:
    """Collate all traces between a date range specified by filters, to get round Langfuse's
    limit on the number of traces it returns"""  # noqa

    filters = ultimate_filters.copy()
    delta = filters["to_timestamp"] - filters["from_timestamp"]

    logger.info("Collating traces ...")

    traces = []

    ct = 0
    while filters["to_timestamp"] <= ultimate_filters["to_timestamp"]:

        if ct % 10 == 0:
            logger.info("\t", ct)

        traces_ = langfuse.fetch_traces(limit=LANGFUSE_TRACE_LIMIT, **filters)
        traces_ = traces_.data

        # print(f'{ct}: {datetime.strftime(filters["from_timestamp"],"%Y-%m-%d %H:%M:%S")} to {datetime.strftime(filters["to_timestamp"],"%Y-%m-%d %H:%M:%S")} ({delta}): {len(traces_)} results') # noqa

        if len(traces_) == LANGFUSE_TRACE_LIMIT:
            delta = delta / 2
            filters = {"from_timestamp": filters["from_timestamp"], "to_timestamp": filters["from_timestamp"] + delta}

        elif len(traces_) <= 5:
            delta = delta * 1.5
            filters = {"from_timestamp": filters["from_timestamp"], "to_timestamp": filters["from_timestamp"] + delta}

        else:
            traces += traces_
            filters = {
                "from_timestamp": filters["from_timestamp"] + delta,
                "to_timestamp": filters["from_timestamp"] + 2 * delta,
            }

        ct += 1

    return traces


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

langfuse = Langfuse()

filters = {
    "from_timestamp": datetime.strptime("2024-12-05 00:00:00", "%Y-%m-%d %H:%M:%S"),
    "to_timestamp": datetime.strptime("2025-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
}


traces = collate_traces(filters)


traces = [trace for trace in traces if not is_admin(trace)]
logger.info(f"N traces = {len(traces)}")

samples, trace_ids = traces_to_samples(traces)

# for i,trace in enumerate(traces):
#   print(f'{i+1}/{len(traces)}',trace.input["input"])
#  input()

skip = int(sys.argv[1]) if sys.argv[1:] else 0

new_sample_marker = "--------------------------------------------"
for i, sample in enumerate(samples[skip:]):

    trace_id = trace_ids[skip + i]
    trace = langfuse.fetch_trace(trace_id).data

    context = "\n".join(
        [
            f"\t{doc['metadata']['title']} ({doc['metadata']['date_pub']}) {doc['metadata']['location']}"
            for doc in trace.output["context"]
        ]
    )

    logger.info(
        f'\n\n\n{new_sample_marker}\n{i+skip+1}/{len(samples)} {bold(trace_id)} {bold(f"[{sample.__class__.__name__}]")} {trace.timestamp} {sample.pretty_repr()}\n\n{context}'  # noqa
    )

    if yesno("\nAdd tags?:"):

        answer = input("List tags separated by spaces: ")
        if answer:
            tags = re.split(r"\s+", answer.strip())
            if tags:
                langfuse.trace(id=trace_id).update(
                    tags=tags, timestamp=trace.timestamp
                )  # it will change the timestamp unless you deliberately keep it the same
