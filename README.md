# Langchain experiments phase I

**LLM call caching**

### Setup

Python 3.10 and a virtualenv with the `requirements.txt`. Also create your
own `.env` based on the template (you will need an OpenAI API key).

### Plan

First reproduce parts of the
["How to cache LLM calls"](https://python.langchain.com/en/latest/modules/models/llms/examples/llm_caching.html)
page (some storage backends at least).

Then work an Astra/Cassandra version.

Then package it into a neat tutorial.

## Reproduce

This is `01_other-backends`.

Tried in-memory, SQLite: all as expected.

Also tried the other backends (keep reading).

#### Redis

To try Redis, a modern Redis (with "RediSearch" module enabled) is required to run,
see [here](https://redis.io/docs/stack/search/quick_start/).

So in a console start this:

```
docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
```

Once the redis cache is used, one sees this in the Redis log: `<search> creating vector index. Server memory limit: [...]`.
All good then.

#### GPTCache

A third-party package that integrates with LangChain to work at the
embedding/vectorsearch layer for retrieval of cached (and not only).

How this works: for each service (such as `openai`) GPTCache offers "wrappers",
i.e. monkeypatches to the service which add the caching behaviour + pass-through everything else.

See [this](https://gptcache.readthedocs.io/en/latest/_modules/gptcache/adapter/openai.html#ChatCompletion), [this](https://github.com/zilliztech/GPTCache/blob/6a1e2e82aabcd3a48486042ef5c7c6323f8589fd/gptcache/adapter/adapter.py#L8) and [this](https://github.com/zilliztech/GPTCache/blob/6a1e2e82aabcd3a48486042ef5c7c6323f8589fd/gptcache/adapter/openai.py#L73-L85).

GPTCache can work in exact-match mode or (when equipped with embedding and suitable search engine/matching algorithms)
in semantic-similarity mode.

Exact mode: in the notebook the "backend" for the cache is a file
(which gets flushed at some point beyond our control, see
[here](https://github.com/zilliztech/GPTCache/blob/6a1e2e82aabcd3a48486042ef5c7c6323f8589fd/gptcache/manager/data_manager.py#L68).
It's a pickle with little more than a simple dict in it, a LRU cache, rather dev-mode.)

Semantic similarity mode: some tools are stuffed into the "cacher" for the LangChain thing, and then it just works.
In this case a local sqlite (out of several SQL backends) is used.

#### SQLAlchemy Cache

The base example there is commented for some reason.

##### A MySQL failure

Let's try it with MySQL: first,

```
docker run --name local-mysql -e MYSQL_ROOT_PASSWORD=123456 -d mysql
```

then [more steps](https://www.howtogeek.com/devops/how-to-run-mysql-in-a-docker-container/) are to be done.

Log to the MySQL with `docker exec -it local-mysql mysql -p` and then in the mysql shell run:

```
create database CACHEDB;
create user 'cacher'@'%' identified by 'cachepwd';
grant all privileges on CACHEDB.* TO 'cacher'@'%';
flush privileges;
```

For Python, some massaging to get MySQL working

```
sudo apt-get install libmysqlclient-dev
sudo apt-get install python-dev default-libmysqlclient-dev
sudo apt install libffi-dev
sudo apt install python3.10-dev
pip install mysqlclient
```

or alternatively just

```
pip install pymysql
```

then change the connection string in the Python cell to either of:

```
# engine = create_engine("mysql+mysqldb://cacher:cachepwd@172.17.0.2/CACHEDB")
engine = create_engine("mysql+pymysql://cacher:cachepwd@172.17.0.2/CACHEDB")
```

But **FAILURE**, it seems the SQLAlchemy fails in required varchar length in this sql dialect:

```
(in table 'full_llm_cache', column 'prompt'): VARCHAR requires a length on dialect mysql
```

Too much time wasted already. Erase the container and try with another SQL DB.

##### Another SQL DB failure

MariaDB, let's see.

```
docker run -p 127.0.0.1:3306:3306  --name mdb -e MARIADB_ROOT_PASSWORD=Password123! -d mariadb:latest
docker exec -it mdb mariadb --user root -pPassword123!
# ... plus the same user/db creation as for mysql.
```

Connection string, see [here](https://github.com/zilliztech/GPTCache/blob/6a1e2e82aabcd3a48486042ef5c7c6323f8589fd/gptcache/manager/scalar_data/sql_storage.py#L99) for a survey:

```
engine = create_engine("mariadb+pymysql://cacher:cachepwd@localhost/CACHEDB")
```

Sadly, **same problem**.

#### PostgreSQL

First,

```
sudo apt install libpq-dev
pip install psycopg2
```

```
docker run --name some-postgres -e POSTGRES_PASSWORD=cachepwd -d postgres
```

This works without complaining about the VARCHAR stuff:

```
engine = create_engine("postgresql://postgres:cachepwd@172.17.0.2:5432/postgres")
```

#### SQLite (meh)

No time to explore further now, switching to SQLite:

```
engine = create_engine("sqlite:///./sqlite.db")
```

This works

#### Notes on existing backends:

In-mem and SQLite implement exact string match.

Better approach: `RedisSemanticCache`. It uses a vector search on embeddings of the prompt.

## Astra/Cassandraify

#### Thoughts

Plug a pretend-bruteforce vector thing, how, how sophisticated and at which level?

SQLAlchemy is exact-match

##### GPTCache

It might be useful to work on extending [GPTCache](https://github.com/zilliztech/GPTCache)

Still in GPTCache, in particular the `CacheBase` [class](https://github.com/zilliztech/GPTCache/blob/6a1e2e82aabcd3a48486042ef5c7c6323f8589fd/gptcache/manager/scalar_data/manager.py#L17) supports a handful of SQL storage engines to save the (similarity) cache state. Worth extending to C* ?


## Cassandraify

This is in `02_cassandra-astra` and we start with `Cassandra_01` notebook.

First a bit of research into the [caching interface](https://github.com/hwchase17/langchain/blob/master/langchain/cache.py).

Then a first prototype is out: a class `CassandraCache` with:

- a cql Session object ready to use and a keyspace name when initialized;
- store/get/clear methods;
- a serdes layer for the predictions.

Not (yet) in scope:

  - handling of TTL
  - custom table name (and CL, TTL policy ...)
  - binary blobs instead of strings for the predictions
  - prepared statements to optimize repeated usage
