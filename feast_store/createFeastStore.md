MOVED TO CASSIO WEBSITE

# Create the Feast store

This is the Feast store needed to run exactly the notebooks that use Feast.
The following sample feature store is done on an Astra DB database;
feel free to use any backend (adapting these instructions to your case);
the LangChain-related part would not need to be modified.

It is required that `feast` is installed in your Python environment.

### Prerequisites

You need an Astra DB database with a keyspace to host tables, a
corresponding Token and the Secure Connect Bundle
(see [here](https://awesome-astra.github.io/) for details).

### Create the store

In this directory (`feast_store`), launch the command:

```
.../feast_store$ feast init -t cassandra user_features
```

choose Astra DB and provide the other required information. You can skip the optional parameters altogether.

Now the `user_features` store is created.

```
$> cd user_features/feature_repo/
```

Prepare sample data in Parquet format:

```
$> python ../../prepare_feast_data.py
```

Replace the example feature definitions file with the provided one, which
matches the sample data and the examples in the notebooks:

```
$> rm example_repo.py 
$> cp ../../user_data_feature_definitions.py .
```

Now initialize the backend: after running the command

```
$ feast -apply
```

you should see (empty) tables in Astra DB.

Now launch the materialization,

```
DATE0=$(date -d "`date` - 10 years" "+%Y-%m-%dT%H:%M:%S")
DATE1=`date "+%Y-%m-%dT%H:%M:%S"`
feast materialize $DATE0 $DATE1
```

and the tables will be written with the data from the Parquet sources.

You are good to run the Feast-related notebooks.
