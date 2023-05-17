# Google's VertexAI LLM setup

(pending Google's integration being merged into LangChain)

The instructions here refer to a "google cloud aiplatform Python wheel" file that is presumably not public yet. It is referenced in the colab about integrating VertexAI and LangChain. Get that `gs://...` url for later.

We also assume you do have a Google account and that it is granted access to VertexAI: you will need to log in to that account.

## Setup

Create a virtualenv and `pip install` the `requirements.txt`.

Check you do `export GOOGLE_CLOUD_PROJECT="<digits, digits...>"` in your `.env` and source it in the shell where you'll start Jupyter.

You need to install `gcloud` (e.g. via snap) on your machine.

As suggested [here](https://cloud.google.com/docs/authentication/provide-credentials-adc), dump a set of valid credentials in your machine with:

```
gcloud auth application-default login
```

(you will see a browser window. Log in to your Google account).

Now,

```
pip install langchain google-cloud-core google-cloud-logging
```

Then,

```
gcloud auth login
```

and also (fill in your Project ID)

```
gcloud config set project <digits, digits ...>
```

Now the following command will work, to get the Python wheel for Google's AI platform:

```
gsutil cp gs://ADDRESS_OF_THE_WHEEL_FILE .
```

Check you have the file and finally install it:

```
pip install invoke transformers
pip install BLABLABLA.whl "shapely<2.0.0"
```

**Optional** Install the vertex-search-enabled Python drivers for Cassandra, if you want to use those features, with:

```
pip install git+https://github.com/datastax/python-driver.git@cep-vsearch#egg=cassandra-driver
```

(if something goes wrong, check the [manual-install instructions](https://github.com/jbellis/cassgpt#dependencies) as well, section about the Python drivers).

## Starting the notebooks

After `. .env`, you should simply launch `jupyter notebook` and wait for a browser window to open.
