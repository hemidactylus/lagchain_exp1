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


class CassandraSemanticCache(BaseCache):
    """
    Cache that uses Cassandra as a vector-store backend,
    based on the CEP-30 drafts at the moment.
    """

    # TODOs:
    #   1. TTL handling (and interface for it in the class)
    #   2. deletion of an index (= one model_str). Hence, concurrency in
    #      caching access under whole-index deletion
    #   3. whether to have the vector as primary key (odd) or an ID
    #   4. other metrics than dot-product ?
    #   5. tune threshold for tolerance
    #   6. implement 'clear()'
    #   7. raise the number of retrieved rows (at the moment this might break the code where there are fewer rows on the table)

    def __init__(
        self, session: Session, keyspace: str,
        embedding: Embeddings, score_threshold: float = SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD,
        distance_metric: str = 'dot',
    ):
        """Initialize the cache with all relevant parameters.
        Args:
            session (cassandra.cluster.Session): an open Cassandra session
            keyspace (str): the keyspace to use for storing the cache
            embedding (Embedding): Embedding provider for semantic encoding and search.
            score_threshold (float, 0.2)
            distance_metric (str, 'dot')
        """
        self.session = session
        self.keyspace = keyspace
        self.embedding = embedding
        self.score_threshold = score_threshold
        self.distance_metric = distance_metric
        if self.distance_metric != 'dot':
            raise NotImplementedError(f'Distance metric "{self.distance_metric}" not supported')
        #
        self._num_rows_to_fetch = SEMANTIC_CACHE_DEFAULT_NUM_ROWS_TO_FETCH
        # we need to keep track of tables/indexes we created already
        # TODO: the following approach is not ready to handle concurrent access (re whole-index deletion)
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

    @staticmethod
    def _get_distances(embeddings, referenceEmbedding):
        """
        Given a list [emb_i] and a reference rEmb vector,
        return a list [distance_i] where each distance is
            distance_i = distance(emb_i, rEmb)
        At the moment only the dot product is supported
        (which for unitary vectors is the cosine difference).

        Not particularly optimized.
        """
        return list(np.dot(
            np.array(embeddings, dtype=float),
            np.array(referenceEmbedding, dtype=float),
        ))

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
        # here we should drop the table and the index, or at least truncate the table.
        # Note: 'truncate if exists' ...
        raise NotImplementedError

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        self._ensureTableExists(llm_string)
        print('CACHING_PROMPT "%s" ==> "%s"' % (prompt, str(return_val)))
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
                CassandraSemanticCache._get_distances(row_embeddings, promptEmbedding),
                rows,
            ))
            #
            print('LOOKUP_TOP_SCORES: %s' % (
                ', '.join('%.5f' % x for x in sorted([s for s, _ in rows_with_metric], reverse=True)[:5])
            ))
            # sort rows by metric score
            sorted_passing_winners = sorted(
                (
                    pair
                    for pair in rows_with_metric
                    if pair[0] >= self.score_threshold
                ),
                key=itemgetter(0),
                reverse=True,
            )
            if sorted_passing_winners:
                # we have a winner. Unpack the pair and rehydrate the generations
                max_score, best_row = sorted_passing_winners[0]
                print('CACHED_MATCH_FOUND: %.5f' % max_score)
                return deserializeGenerationsToString(best_row.generations_str)
            else:
                # no row passes the threshold score
                return None
        else:
            # no rows returned by the ANN search
            return None
