# LangChain experiments

With the goal of providing Cassandra / Astra DB support in the form of
classes, practices and extensions.

General note: **notebooks** show how to use, **py-files** implement the tools.

Some of the notebooks below work with either OpenAI's or Google's choice
of LLM. In the notebook this shows as a cell with a simple binary choice;
but, whereas for OpenAI you just need to provide a valid `OPENAI_API_KEY`
in `.env`, for Google (running locally) you have to go through a more
convoluted setup, described in a [separate document](google_setup.md).

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

## Phase VI - Prompt Templates

A high level of abstraction to seamlessly inject data from Cassandra into a prompt.

Some engineering was required, which worked in two steps:

1. a generic prompt template able to cope with dependencies and multiple arguments to the data 'getters'
2. a specialization that just requires declaratively describing which tables contribute with their columns.

The classes need some work, mostly in adding all sorts of validation steps re: overlapping arg names and such.

#### A sub-topic: Feast integration

This expands on the [Feast](https://python.langchain.com/en/latest/modules/prompts/prompt_templates/examples/connecting_to_a_feature_store.html#feast) integration example on LangChain site.

First we set up a minimal Feast,
then we try the vanilla integration, then we see what to do to improve/automate things as done above.

We will use an Astra-backed Feast installation :)

Follow these [instructions](feast_store/createFeastStore.md) and get a feature store ready for the next notebooks.

Then, `02_cassandra-astra/feastDIY-Cassandra_01.ipynb` for an example just like in the tutorial.

For Feast, the same simplification was done as for Cassandra, compare the above
with the `feastPromptTemplate-Cassandra_01` notebook.

#### A sub-topic: Chat Prompt Templates

A facility to handle (and propagate `format` arguments through)
hierarchies of "messages" by human/AI/system (or other)
in a single prompt, talored to become the grand prompt
to the LLM. Also supports output formats other than "string".

Testing how the above "cassandra Prompt Templates" can be used
within these higher-level chains (sequences) of string prompts,
looks like you can.

Experimenting in `02_cassandra-astra/chatprompttemplates-Cassandra_01`.

It seems it works nicely.