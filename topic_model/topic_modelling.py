from __future__ import annotations

from collections import OrderedDict
from typing import List
from typing import Optional

import lancedb
import numpy as np

from config import DB_PATH
from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger

# from dsp_nesta_brain import logger
from hdbscan import HDBSCAN
from retrieval.db.schema.policy_atlas import Activity
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


# might need for cluster labelling:
# from sklearn.feature_extraction.text import TfidfVectorizer
# result = tfidf.fit_transform(s)
# https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/

# based on: https://ai.plainenglish.io/revolutionizing-topic-modeling-with-gpt-3-5-from-text-embedding-to-contextual-titles-1b9fa187b76b  # noqa

DATA_PATH = PROJECT_DIR / "data/policy_atlas/fcdo_iati_data_2025_01_17.csv"


class Cluster:
    """Topic modelling cluster identified by HDBSCAN"""

    label: int
    activities: List[Activity]
    similarity_matrix_: Optional[List[List[float]]] = None

    def __init__(self, label: int) -> None:
        self.label = label
        self.activities = []

    @staticmethod
    def hdbscan_to_Clusters(hdbscan_model_results: HDBSCAN, activities: List[Activity]) -> List[Cluster]:
        """Derive a list of Clusters from the results of HDBSCAN"""

        clusters = {}
        for i, label in enumerate(hdbscan_model_results.labels_):
            if label not in clusters:
                clusters[label] = Cluster(label)
            clusters[label].activities.append(activities[i])

        clusters = OrderedDict({k: clusters[k] for k in sorted(clusters.keys())})
        return list(clusters.values())

    @property
    def embeddings(self) -> List[List[float]]:
        """Return the embeddings belonging to the cluster activities"""
        return [activity.vector for activity in self.activities]

    @property
    def representative_activity(self) -> Activity:
        """Return a single activity which is representative of the cluster"""
        if len(self.activities) > 1:
            sim_sum = self.similarity_matrix.sum(axis=1)
            activity_index = np.argmax(sim_sum)
            return self.activities[activity_index]
        return self.activities[0]

    @property
    def similarity_matrix(self) -> List[List[float]]:
        """Return a matrix presenting the cosine similarities of each activity with every other activity in the cluster"""
        if self.similarity_matrix_ is None:
            self.similarity_matrix_ = cosine_similarity(self.embeddings, self.embeddings)
        return self.similarity_matrix_

    @property
    def size(self) -> int:
        """Return the number of activities in the cluster"""
        return len(self.activities)


if __name__ == "__main__":

    db = lancedb.connect(DB_PATH)

    table = db.open_table("activity")
    activities = table.search().limit(100).to_pydantic(Activity)
    embeddings = [activity.vector for activity in activities]

    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric="cosine")
    hdbscan_model = HDBSCAN(min_cluster_size=3, metric="euclidean", min_samples=2, prediction_data=False)

    reduced_embeddings = umap_model.fit_transform(embeddings)
    hdbscan_model_results = hdbscan_model.fit(reduced_embeddings)
    clusters = Cluster.hdbscan_to_Clusters(hdbscan_model_results, activities)

    for cluster in clusters:

        if cluster.label == -1:
            continue

        #   if cluster.size <= top_n:
        #      continue

        logger.info(f"{cluster.label}.\n{cluster.representative_activity}\n")
