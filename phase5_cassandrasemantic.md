# Phase 5: Cassandra semantic cache

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
