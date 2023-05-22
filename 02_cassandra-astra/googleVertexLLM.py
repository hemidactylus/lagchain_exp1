"""
A very-alpha class to instantiate a Google LLM as a drop-in replacement
for the OpenAI LLM, ready to be used in all LangChain notebooks.
"""

import time
import logging
import os
import json

from vertexai.preview import language_models
from langchain.embeddings.base import Embeddings
from google.cloud import aiplatform
from google.oauth2 import service_account
import google.cloud.logging

from langchain.llms.base import LLM
# Imports the Cloud Logging client library
# Imports Python standard library logging
# from langchain import PromptTemplate, LLMChain

GOOGLE_CLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
GOOGLE_AUTH_JSON_PATH = os.environ['GOOGLE_AUTH_JSON_PATH']


json_str = open(GOOGLE_AUTH_JSON_PATH).read()
json_data = json.loads(json_str)
json_data['private_key'] = json_data['private_key'].replace('\\n', '\n')

credentials = service_account.Credentials.from_service_account_info(json_data)
aiplatform.init(project=GOOGLE_CLOUD_PROJECT, credentials=credentials)

#
# Create a child class of the base Langchain LLM base class for Vertex LLM API
#
class VertexLLM(LLM):

 # model: language_models.LanguageModel (old code which was broken)
  model: language_models.TextGenerationModel
  predict_kwargs: dict
  req_logging: bool
  model_name: str

  DEFAULT_LLM_LOG_NAME = "vertex-llm-langchain-log"

  def __init__(self, model_name = "text-bison-001", req_logging = False, **predict_kwargs):
    vertexmodel = language_models.TextGenerationModel.from_pretrained(model_name)
    super().__init__(model=vertexmodel, model_name=model_name, req_logging=req_logging, predict_kwargs=predict_kwargs)

  @property
  def _llm_type(self):
    return 'vertex'

  def _call(self, prompt, stop=None):

    result = self.model.predict(prompt, **self.predict_kwargs)

    #
    # If req_logging = True,
    # log all requests-response to the Vertex LLM API to Cloud Logging
    #
    # TODO: Figure out how to have more advanced tracing of chains of LLM calls
    #       like https://python.langchain.com/en/latest/tracing.html
    #
    if self.req_logging :
      logging_client = google.cloud.logging.Client()
      logging_client.setup_logging()
      log_name = self.DEFAULT_LLM_LOG_NAME
      logger = logging_client.logger(log_name)
      logger.log_struct(
      {
          "prompt": prompt,
          "reponse": str(result),
          "model": {
              "name" : self.model_name,
              "MAX_OUTPUT_TOKENS" : self.predict_kwargs['max_output_tokens'],
              "TEMPERATURE" : self.predict_kwargs['temperature'],
              "TOP_P" : self.predict_kwargs['top_p'],
              "TOP_K" : self.predict_kwargs['top_k']
          }
      })
    return str(result)

  @property
  def _identifying_params(self):
    return {}

def rate_limit(max_per_minute):
  period = 60 / max_per_minute
  while True:
    before = time.time()
    yield
    after = time.time()
    elapsed = after - before
    sleep_time = max(0, period - elapsed)
    if sleep_time > 0:
      print(f'Sleeping {sleep_time:.1f} seconds')
      time.sleep(sleep_time)

#
# Create a child class of the base Langchain Embedding base class for
# Vertex PaLM Embedding API
#
class VertexEmbeddings(Embeddings):

  def __init__(self, model, *, requests_per_minute=15):
    self.model = model
    self.requests_per_minute = requests_per_minute

  def embed_documents(self, texts):
    limiter = rate_limit(self.requests_per_minute)
    results = []
    docs = list(texts)

    while docs:
      # Working in batches of 2 because the API apparently won't let
      # us send more than 2 documents per request to get embeddings.
      head, docs = docs[:2], docs[2:]
      chunk = self.model.get_embeddings(head)
      results.extend(chunk)
      next(limiter)

    return [r.values for r in results]

  def embed_query(self, text):
    single_result = self.embed_documents([text])
    return single_result[0]


def googleVertexAILLM():
  #Initialise a new instance of Vertex LLM Object for use
  vertexQnALLMInstance = VertexLLM(
    req_logging=True,
    model_name='text-bison@001',
    max_output_tokens=512,
    temperature=0,
    top_p=0.8,
    top_k=40
  )
  return vertexQnALLMInstance

def googleVertexEmbeddings():
  embedding_model = language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
  vertexEmbeddings = VertexEmbeddings(
    model=embedding_model,
  )
  return vertexEmbeddings
