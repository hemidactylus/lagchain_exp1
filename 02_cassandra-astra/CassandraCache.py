"""Prototype of an exact-search cache for LangChain using Cassandra / Astra DB."""

from operator import itemgetter
from functools import lru_cache
import hashlib
import json
from typing import Any, Optional, List
import numpy as np

from langchain.cache import BaseCache, RETURN_VAL_TYPE
from langchain.schema import Generation
from langchain.embeddings.base import Embeddings

try:
    from cassandra.cluster import Session
    from cassandra.query import SimpleStatement
except ImportError:
    raise ValueError(
        "Could not import cassandra python package. "
        "Please install it with `pip install cassandra-driver`."
    )


# CQL templates
_createTableCQLTemplate = """
CREATE TABLE IF NOT EXISTS {keyspace}.{tableName} (
    llm_string TEXT,
    prompt TEXT,
    generations_str TEXT,
    PRIMARY KEY (( llm_string, prompt ))
);
"""
_getCachedItemCQLTemplate = """
SELECT generations_str
    FROM {keyspace}.{tableName}
WHERE llm_string=%s
    AND prompt=%s;
"""
_storeCachedItemCQLTemplate = """
INSERT INTO {keyspace}.{tableName} (
    llm_string,
    prompt,
    generations_str
) VALUES (
    %s,
    %s,
    %s
);
"""
_truncateTableCQLTemplate = """
TRUNCATE TABLE {keyspace}.{tableName};
"""

# CQL templates for the semantic cache
_createSemanticTableCQLTemplate = """
CREATE TABLE IF NOT EXISTS {keyspace}.{tableName} (
    prompt_id TIMEUUID PRIMARY KEY,
    embedding FLOAT VECTOR[{embeddingDimension}],
    prompt TEXT,
    generations_str TEXT
);
"""
_createSemanticIndexCQLTemplate = """
CREATE CUSTOM INDEX IF NOT EXISTS {indexName} ON {keyspace}.{tableName} (embedding)
USING 'org.apache.cassandra.index.sai.StorageAttachedIndex' ;
"""
_storeCachedSemanticItemCQLTemplate = """
INSERT INTO {keyspace}.{tableName} (
    prompt_id,
    embedding,
    prompt,
    generations_str
) VALUES (
    now(),
    %s,
    %s,
    %s
);
"""
_getCachedSemanticItemCQLTemplate = """
SELECT
    embedding, generations_str
FROM {keyspace}.{tableName}
    WHERE embedding ANN OF %s
    LIMIT %s
"""
_truncateSemanticTableCQLTemplate = """
TRUNCATE TABLE {keyspace}.{tableName};
"""


SEMANTIC_CACHE_DEFAULT_NUM_ROWS_TO_FETCH = 1
SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE = 16
SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD = 0.85

# hashing function for model string -> valid identifier
def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


# String serdes layer
def serializeGenerationsToString(generations: RETURN_VAL_TYPE) -> str:
    return json.dumps([
        gen.text
        for gen in generations
    ])


def deserializeGenerationsToString(blob_str: str) -> RETURN_VAL_TYPE:
    return [
        Generation(text=txt)
        for txt in json.loads(blob_str)
    ]


class CassandraCache(BaseCache):
    """
    Cache that uses Cassandra / Astra DB as a backend.

    It uses a single table. The lookup keys (also primary keys) are:
        - prompt, a string
        - llm_string, a string deterministic representation of the model parameters.
          This is to keep collision between same prompts for two models separate.
    """

    def __init__(self, session: Session, keyspace: str):
        """Initialize with a ready session and a keyspace name."""
        if not isinstance(session, Session):
            raise ValueError("Please provide a Session object.")
        self.session = session
        if not isinstance(keyspace, str):
            raise ValueError("Please specify a working keyspace.")
        self.keyspace = keyspace
        self.tableName = 'langchain_cache'
        # Schema creation, if needed
        createTableCQL = _createTableCQLTemplate.format(
            keyspace=self.keyspace,
            tableName=self.tableName,
        )
        session.execute(createTableCQL)
        #

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        getCachedItemCQL = _getCachedItemCQLTemplate.format(
            keyspace=self.keyspace,
            tableName=self.tableName,
        )
        foundItem = self.session.execute(
            getCachedItemCQL,
            (llm_string, prompt),
        ).one()
        if foundItem:
            return deserializeGenerationsToString(foundItem.generations_str)
        else:
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        storeCachedItemCQL = _storeCachedItemCQLTemplate.format(
            keyspace=self.keyspace,
            tableName=self.tableName,
        )
        self.session.execute(
            storeCachedItemCQL,
            (llm_string, prompt, serializeGenerationsToString(return_val)),
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        truncateTableCQL = _truncateTableCQLTemplate.format(
            keyspace=self.keyspace,
            tableName=self.tableName,
        )
        self.session.execute(
            truncateTableCQL
        )


# distance definitions. These all work batched in the first argument.
def distance_cosDifference(embeddings: List[List[float]], referenceEmbedding: List[float]) -> List[float]:
    v1s = np.array(embeddings, dtype=float)
    v2 = np.array(referenceEmbedding, dtype=float)
    return list(np.dot(
        v1s,
        v2.T,
    ) / (
        np.linalg.norm(v1s, axis=1)
        * np.linalg.norm(v2)
    ))


def distance_dotProduct(embeddings: List[List[float]], referenceEmbedding: List[float]) -> List[float]:
        """
        Given a list [emb_i] and a reference rEmb vector,
        return a list [distance_i] where each distance is
            distance_i = distance(emb_i, rEmb)
        At the moment only the dot product is supported
        (which for unitary vectors is the cosine difference).

        Not particularly optimized.
        """
        v1s = np.array(embeddings, dtype=float)
        v2 = np.array(referenceEmbedding, dtype=float)
        return list(np.dot(
            v1s,
            v2.T,
        ))


def distance_L1(embeddings: List[List[float]], referenceEmbedding: List[float]) -> List[float]:
        v1s = np.array(embeddings, dtype=float)
        v2 = np.array(referenceEmbedding, dtype=float)
        return list(np.linalg.norm(v1s - v2, axis=1, ord=1))


def distance_L2(embeddings: List[List[float]], referenceEmbedding: List[float]) -> List[float]:
        v1s = np.array(embeddings, dtype=float)
        v2 = np.array(referenceEmbedding, dtype=float)
        return list(np.linalg.norm(v1s - v2, axis=1, ord=2))


def distance_max(embeddings: List[List[float]], referenceEmbedding: List[float]) -> List[float]:
        v1s = np.array(embeddings, dtype=float)
        v2 = np.array(referenceEmbedding, dtype=float)
        return list(np.linalg.norm(v1s - v2, axis=1, ord=np.inf))


# function, sorting (True = higher is closer)
distanceMetricsMap = {
    'cos': (
        distance_cosDifference,
        True,
    ),
    'dot': (
        distance_dotProduct,
        False,
    ),
    'l1': (
        distance_L1,
        False,
    ),
    'l2': (
        distance_L2,
        False,
    ),
    'max': (
        distance_max,
        False,
    ),
}


class CassandraSemanticCache(BaseCache):
    """
    Cache that uses Cassandra as a vector-store backend,
    based on the CEP-30 drafts at the moment.

    - TTL is not supported yet
    - As soon as cassandra admist LIMIT > total rows in table (alpha crashes now),
      raise SEMANTIC_CACHE_DEFAULT_NUM_ROWS_TO_FETCH to 32 or so.
    """

    def __init__(
        self, session: Session, keyspace: str,
        embedding: Embeddings,
        distance_metric: str = 'dot',
        score_threshold: float = SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD,
    ):
        """Initialize the cache with all relevant parameters.
        Args:
            session (cassandra.cluster.Session): an open Cassandra session
            keyspace (str): the keyspace to use for storing the cache
            embedding (Embedding): Embedding provider for semantic encoding and search.
            distance_metric (str, 'dot')
            score_threshold (optional float)
        The default score threshold is tuned to the default metric.
        Tune it carefully yourself if switching to another distance metric.
        """
        self.session = session
        self.keyspace = keyspace
        self.embedding = embedding
        self.score_threshold = score_threshold
        self.distance_metric = distance_metric
        if self.distance_metric not in distanceMetricsMap:
            raise NotImplementedError(f'Distance metric "{self.distance_metric}" not supported')
        #
        self._num_rows_to_fetch = SEMANTIC_CACHE_DEFAULT_NUM_ROWS_TO_FETCH
        # we keep track of tables/indexes we created already, to avoid doing
        # a lot of 'create if not exists' all the time
        self.table_cache = {}  # model_str -> table_name
        # The contract for this class has separate lookup and update:
        # in order to spare some embedding calculations we cache them between
        # the two calls.
        # Note: the `_cache_embedding` copies in two instances of this cache
        # class will be two separate items, each handled separately by the lru.
        @lru_cache(maxsize=SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        def _cache_embedding(text):
            return self.embedding.embed_query(text=text)
        self._get_embedding = _cache_embedding

    def _ensureTableExists(self, llm_string):
        """
        Create the table and the index if they don't exist.
        This is twice expensive (runs an embedding and does DML on Cassandra),
        and that's why we keep a local cache of what's been created already.
        """
        if llm_string not in self.table_cache:
            tableName = CassandraSemanticCache._getTableName(llm_string)
            vectorIndexName = CassandraSemanticCache._getVectorIndexName(llm_string)
            embeddingDimension = self._getEmbeddingDimension()
            # create table and index with statements
            createTableCQL = SimpleStatement(_createSemanticTableCQLTemplate.format(
                keyspace=self.keyspace,
                tableName=tableName,
                embeddingDimension=embeddingDimension,
            ))
            createIndexCQL = SimpleStatement(_createSemanticIndexCQLTemplate.format(
                indexName=vectorIndexName,
                keyspace=self.keyspace,
                tableName=tableName,
            ))
            self.session.execute(createTableCQL)
            self.session.execute(createIndexCQL)
            self.table_cache[llm_string] = tableName
        else:
            # our local cache tells us we already created this table: no-op
            pass

    def _get_distances(self, embeddings: List[List[float]], referenceEmbedding: List[float]) -> List[float]:
        metric_function = distanceMetricsMap[self.distance_metric][0]
        return metric_function(embeddings, referenceEmbedding)

    @staticmethod
    def _getTableName(llm_string):
        return f'semantic_cache_{_hash(llm_string)}'

    @staticmethod
    def _getVectorIndexName(llm_string):
        return f'{CassandraSemanticCache._getTableName(llm_string)}_embedding_idx'

    def _getEmbeddingDimension(self):
        return len(self._get_embedding(text="test"))

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        if 'llm_string' not in kwargs:
            raise ValueError('llm_string parameter must be passed to clear()')
        else:
            llm_string = kwargs["llm_string"]        
            tableName = CassandraSemanticCache._getTableName(llm_string)
            truncateTableCQL = _truncateSemanticTableCQLTemplate.format(
                keyspace=self.keyspace,
                tableName=tableName,
            )
            self.session.execute(
                truncateTableCQL
            )

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        self._ensureTableExists(llm_string)
        # calculate values to insert
        tableName = CassandraSemanticCache._getTableName(llm_string)
        generations_str = serializeGenerationsToString(return_val)
        embedding = self._get_embedding(text=prompt)
        storeCachedItemCQL = SimpleStatement(_storeCachedSemanticItemCQLTemplate.format(
            keyspace=self.keyspace,
            tableName=tableName,
        ))
        self.session.execute(
            storeCachedItemCQL,
            (
                embedding,
                prompt,
                generations_str,
            ),
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        self._ensureTableExists(llm_string)
        # Prepare to fetch a number of rows
        tableName = CassandraSemanticCache._getTableName(llm_string)
        getCachedItemCQL = SimpleStatement(_getCachedSemanticItemCQLTemplate.format(
            keyspace=self.keyspace,
            tableName=tableName,
        ))
        promptEmbedding: List[float] = self._get_embedding(text=prompt)
        # Get ANN rows
        rows = list(self.session.execute(
            getCachedItemCQL,
            (
                promptEmbedding,
                self._num_rows_to_fetch,
            ),
        ))
        if rows:
            # evaluate metric
            row_embeddings = [
                row.embedding
                for row in rows
            ]
            # enrich with their metric score
            rows_with_metric = list(zip(
                self._get_distances(row_embeddings, promptEmbedding),
                rows,
            ))
            # sort rows by metric score
            sorted_passing_winners = sorted(
                (
                    pair
                    for pair in rows_with_metric
                    if pair[0] >= self.score_threshold
                ),
                key=itemgetter(0),
                reverse=distanceMetricsMap[self.distance_metric][1],
            )
            if sorted_passing_winners:
                # we have a winner. Unpack the pair and rehydrate the generations
                max_score, best_row = sorted_passing_winners[0]
                return deserializeGenerationsToString(best_row.generations_str)
            else:
                # no row passes the threshold score
                return None
        else:
            # no rows returned by the ANN search
            return None

if __name__ == '__main__':
    import sys
    # mode = 'ActualEmbeddingTest'
    mode = 'FakeVectorMetricTest'
    #
    if mode == 'ActualEmbeddingTest':
        from langchain.embeddings import OpenAIEmbeddings
        #
        myEmbedding = OpenAIEmbeddings()
        sent1 = 'How does a feed-forward network differ from a recurrent network?'
        _sent2 = ' '.join(sys.argv[1:])
        sent2 = _sent2 + ('?' if(_sent2[-1] != '?') else '')
        print('SENT1 = "%s"' % sent1)
        print('SENT2 = "%s"' % sent2)
        #
        vec1 = myEmbedding.embed_query(sent1)
        vec2 = myEmbedding.embed_query(sent2)
        print('Dot = %.5f' % (
            CassandraSemanticCache._get_distances([vec1], vec2)[0]
        ))
    if mode == 'FakeVectorMetricTest':
        v1a = [1, 2, 3, 4]
        v1b = [1, 0, 0, 1]
        v2 = [-1, 1, 1, -1]
        v1s = [v1a, v1b]
        #
        for mkey, (mfct, srt) in sorted(distanceMetricsMap.items()):
            print('Metric "%s" => %s' % (
                mkey,
                ', '.join('%.5f' % d for d in mfct(v1s, v2))
            ))
