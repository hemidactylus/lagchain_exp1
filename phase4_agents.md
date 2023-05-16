# Phase IV: Agents

_Note: the main purpose of this study is to build an agent-based text-to-CQL-and-back_
_kind of pipeline._

Some exploration on how to create custom agents is in
`01_other-backends/agent-exp-01.ipynb`.

Mostly this is about making sure structured input is accepted
and dependency injection (say, a DB connection, etc) is ok.

Also, packaging a "toolkit" is understood and exemplified.

Far from complete!

#### Cassandra

Trying to come up with a reasonable CQL minimal agent
packaged as a toolkit.

`02_cassandra-astra/cqlagent-Cassandra_01` is the notebook
and `02_cassandra-astra/CQLAgentToolkit.py` is where the implementation is.

This warrants more work...
