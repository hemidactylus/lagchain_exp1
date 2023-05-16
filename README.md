# LangChain experiments

With the goal of providing Cassandra / Astra DB support in the form of
classes, practices and extensions.

General note: **notebooks** show how to use, **py-files** implement the tools.

## Phase I - LLM caching

See [this page](phase1_llmcaching.md).

The reference LangChain page is [here](https://python.langchain.com/en/latest/modules/models/llms/examples/llm_caching.html#).

## Phase II - CQLChain

See [this page](phase2_cqlchain.md).

The reference LangChain page is [here](https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html#).

## Phase III - Memory

See [this page](phase3_memory.md).

The reference LangChain page is [here](https://python.langchain.com/en/latest/modules/memory/getting_started.html#).
Visit the next pages in the navigation as well for memory types built on top.

## Phase IV - Agents

See [this page](phase4_agents.md).

The situation is less well-defined here, with various differing interfaces. I took
inspiration from the source code mostly, plus by reading [this whole section](https://python.langchain.com/en/latest/modules/agents.html).

## Phase V - Cassandra Semantic Cache

Inspired from the `RedisSemanticCache` on [this page](phase1_llmcaching.md) (mostly, inspection of the source code).

See [this page](phase5_cassandrasemantic.md).