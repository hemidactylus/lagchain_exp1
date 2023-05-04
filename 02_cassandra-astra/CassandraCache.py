"""Prototype of an exact-search cache for LangChain using Cassandra / Astra DB."""

import hashlib
import json
from typing import Any, Optional

from langchain.cache import BaseCache, RETURN_VAL_TYPE
from langchain.schema import Generation

try:
    from cassandra.cluster import Session
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
