# LangChain experiments

With the goal of providing Cassandra / Astra DB support in the form of
classes, practices and extensions.

## Phase I - LLM caching

See [this page](phase1_llmcaching.md).

## Phase II - CQLChain

See [this page](phase2_cqlchain.md).

## Phase III - Memory

See [this page](phase3_memory.md)

## Phase IV - Agents

Some exploration on how to create custom agents is in
`01_other-backends/agent-exp-01.ipynb`.

Mostly this is about making sure structured input is accepted
and dependency injection (say, a DB connection, etc) is ok.

Also, packaging a "toolkit" is understood and exemplified.

#### Cassandra

Trying to come up with a reasonable CQL minimal agent
packaged as a toolkit.

`02_cassandra-astra/cqlagent-Cassandra_01` is the notebook
and `02_cassandra-astra/CQLAgentToolkit.py` is where the implementation is.

This warrants more work...

## Phase V - Cassandra Semantic Cache

**NOTE**. This relies on the still-very-alpha CEP-30 Cassandra binaries ("vector capabilities")
and as such should be run with modified Python drivers for Cassandra and a modified Cassandra binary. The setup will be brought to friendlier instructions over time.

We have a separate virtualenv with the special Py driver as dependency.

#### Experiments

These are in `02_cassandra-astra/experiments/vector-cassandra-checks.ipynb`.

The index can return `top_k` ANN rows from the table given an input vector.
For this application, we also need to compute the distance and either keep one
or declare it too distant (hence, nothing cached is usable).

So, since the ANN retrieval is not sorted, we select a certain number and then filter-and-sort
for the semantic cache.

Except for some rough edges and tuning to do, there it is:
`02_cassandra-astra/semantic-llmcache-Cassandra_01.ipynb`
