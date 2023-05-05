# LangChain experiments

With the goal of providing Cassandra / Astra DB support in the form of
classes, practices and extensions.

## Phase I - LLM caching

See [this page](phase1_llmcaching.md).

## Phase II - SQLChain

This is here for the time being.

Reference preprint: [2204.00498](https://arxiv.org/abs/2204.00498).

(SQL) Code: [chain](https://github.com/hwchase17/langchain/blob/8de1b4c4c20ea81f44628a1c42fbc1bbfff37520/langchain/chains/sql_database/base.py#L18) and [prompts](https://github.com/hwchase17/langchain/blob/8de1b4c4c20ea81f44628a1c42fbc1bbfff37520/langchain/chains/sql_database/prompt.py).

### Reproduce

Set up the SQLite chinook database as in the [reference](https://database.guide/2-sample-databases-sqlite/),
so that there's a `Chinook.db` file in the `01_other-backends` directory.
This is where the notebook `sqlchain-Other-Backends-01.ipynb` will run.

#### A possible bug

When doing the "setting limit programmatically", it seems that the closing
semicolon is always lost by the part that adds the `LIMIT n` clause to the query.

A bit of experimentation to have the semicolon.
Still unclear how come it works at the beginning of the notebook (top_k is always there with default = 5).
Possibly another template gets pulled in?

Solution seems that `SQLITE_PROMPT` is never used unless you specify it explicitly.
Anyway this is not enough.

Try with my 'nudged' prompt.

#### A warning

At the time of writing, the internals of the SQLDatabaseSequentialChain still uses a llm
where (so it seems) a llm_chain is needed. A warning ensues (just so we know).

### Cassandraify

A separate notebook just to play with CQL generation in this style: `02_cassandra-astra/experiments-cqlchain-Cassandra`

First we create a sensible `table_info`, then we start with the prompts.

Not implemented stuff, among other:

- indices
- type mapping (?)
- 'with clustering order by'
- sample rows in table_info
- overriding table_info (per-table mapping)

Problems:
- insists on using ORDER BY indiscriminately
- ignores the always-specify-partition request

Very unsatisfactory results.