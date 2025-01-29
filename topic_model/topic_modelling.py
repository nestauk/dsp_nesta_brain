from __future__ import annotations

import os

from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import lancedb
import numpy as np
import openai

from bertopic import BERTopic
from bertopic.representation import OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from config import DB_PATH
from config import DEFAULT_MODEL
from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger

# from dsp_nesta_brain import logger
from hdbscan import HDBSCAN
from retrieval.db.schema.policy_atlas import Activity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


# might need for cluster labelling:
# from sklearn.feature_extraction.text import TfidfVectorizer
# result = tfidf.fit_transform(s)
# https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/

# based on: https://ai.plainenglish.io/revolutionizing-topic-modeling-with-gpt-3-5-from-text-embedding-to-contextual-titles-1b9fa187b76b  # noqa

DATA_PATH = PROJECT_DIR / "data/policy_atlas/fcdo_iati_data_2025_01_17.csv"


def simple_tokenizer(text: str) -> List[str]:
    """Split the text into words"""
    return text.split()


class Cluster:
    """Topic modelling cluster identified by HDBSCAN"""

    label_int: int
    label_str: Optional[str] = None
    activities: List[Activity]
    similarity_matrix_: Optional[List[List[float]]] = None

    def __init__(self, label_int: int, label_str: Optional[str] = None) -> None:
        self.label_int = label_int
        self.label_str = label_str
        self.activities = []

    @staticmethod
    def labels_to_Clusters(
        label_integers: List[int], activities: List[Activity], label_strings: Optional[List[str]] = None
    ) -> List[Cluster]:
        """Derive a list of Clusters from model results"""

        clusters = {}
        for i, label_int in enumerate(label_integers):

            label_str = None
            if label_strings:
                label_str = label_strings[i]

            if label_int not in clusters:
                clusters[label_int] = Cluster(label_int, label_str=label_str)
            clusters[label_int].activities.append(activities[i])

        clusters = OrderedDict({k: clusters[k] for k in sorted(clusters.keys())})
        return list(clusters.values())

    @property
    def embeddings(self) -> List[List[float]]:
        """Return the embeddings belonging to the cluster activities"""
        return [activity.vector for activity in self.activities]

    @property
    def concat_texts(self) -> str:
        """Concatenate all the texts in the cluster"""
        return " ".join([activity.text for activity in self.activities])

    @property
    def label(self) -> str:
        """If no label string exists, create one from keywords"""
        if self.label_str:
            return self.label_str
        return "_".join(self.keywords())

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

    def keywords(
        self,
        n: int = 5,
        vectorizer: Optional[TfidfVectorizer] = None,
    ) -> Dict:
        """
        Generate keywords that characterise the cluster, using the specified Vectorizer

        adapted from cluster_keywords:
        https://github.com/nestauk/discovery_utils/blob/73-refactoring-utils/discovery_utils/utils/viz_landscape.py
        """

        vectorizer = vectorizer or default_vectorizer()

        # Apply the vectorizer
        token_score_matrix = vectorizer.fit_transform([self.concat_texts])

        # Create a token lookup dictionary
        id_to_token = dict(zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys())))

        # Check the top n tokens
        # Get the cluster feature vector
        x = token_score_matrix[0, :].todense()
        # Find the indices of the top n tokens
        x = list(np.flip(np.argsort(np.array(x)))[0])[0:n]
        # Find the tokens corresponding to the top n indices
        top_cluster_tokens = [id_to_token[ele] for ele in x]

        return top_cluster_tokens


def default_vectorizer() -> TfidfVectorizer:
    """Return a default vectorizer"""

    return TfidfVectorizer(
        analyzer="word",
        #  tokenizer=simple_tokenizer,
        preprocessor=lambda x: x,
        #   token_pattern=None,
        #  max_df=0.9,
        # min_df=0.01,
        stop_words="english",
        # max_features=10000,
    )


if __name__ == "__main__":

    model: Literal["HDBSCAN", "BERT"] = "HDBSCAN"

    db = lancedb.connect(DB_PATH)

    table = db.open_table("activity")

    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric="cosine")
    hdbscan_model = HDBSCAN(min_cluster_size=3, metric="euclidean", min_samples=2, prediction_data=False)

    if model == "HDBSCAN":

        activities = table.search().limit(100).to_pydantic(Activity)
        embeddings = [activity.vector for activity in activities]
        reduced_embeddings = umap_model.fit_transform(embeddings)
        hdbscan_model_results = hdbscan_model.fit(reduced_embeddings)
        clusters = Cluster.labels_to_Clusters(hdbscan_model_results.labels_, activities)

    if model == "BERT":

        activities = table.search().limit(100).to_pandas()
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        representation_model = OpenAI(client, model=DEFAULT_MODEL, delay_in_seconds=0.5, chat=True)

        topic_model = BERTopic(
            min_topic_size=5,
            n_gram_range=(1, 1),
            verbose=False,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            umap_model=umap_model,
            representation_model=representation_model,
            nr_topics=10,
        )

        label_integers, _ = topic_model.fit_transform(
            activities["text"],
            embeddings=np.array(activities["vector"].to_list()),
        )

        label_df = topic_model.get_topic_info()
        label_strings = [label_df.loc[label_df["Topic"] == label_int]["Name"] for label_int in label_integers]
        clusters = Cluster.labels_to_Clusters(label_integers, activities, label_strings=label_strings)

    #       print(topic_model.get_topic_info())

    for cluster in clusters:

        if cluster.label_int == -1:
            continue

        #   if cluster.size <= top_n:
        #      continue

        logger.info(f"{cluster.label_int} {cluster.label}.\n{cluster.representative_activity}\n\n")
