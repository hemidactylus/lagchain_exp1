# Phase III: Memory

Implemented in `02_cassandra-astra/CassandraChatMessageHistory.py` and exemplified in `02_cassandra-astra/memory-Cassandra_01`.

General structure: one creates a `CassandraChatMessageHistory` class,
which subclasses `BaseChatMessageHistory`, then [uses](https://github.com/hwchase17/langchain/blob/42df78d3964170bab39d445aa2827dea10a312a7/tests/integration_tests/memory/test_cosmos_db.py#L16) it e.g. as in:

```
message_history = CassandraChatMessageHistory(
    STUFF ...
    ttl=10,
)
memory = ConversationBufferMemory(
    memory_key="<check-template-matching>",
    chat_memory=message_history,
    return_messages=True,
)
...
```

See also [the generic docs on memory](https://python.langchain.com/en/latest/modules/memory/examples/adding_memory.html).

Not in scope for now:

- preparing statements
- better serialization
- clash between `session_id` and (CQL) `session` in the class

Once this is in place, it can be used with other types of memory than the simple `ConversationBufferMemory`.
In `02_cassandra-astra/memory-Cassandra_02-summarybuffermemory` we test it with the `ConversationSummaryBufferMemory`.