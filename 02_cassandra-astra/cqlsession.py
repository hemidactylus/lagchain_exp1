# This is where the session is created to later pass it to the CassandraCache
import os
from dotenv import load_dotenv
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

load_dotenv()

ASTRA_DB_SECURE_BUNDLE_PATH = os.environ["ASTRA_DB_SECURE_BUNDLE_PATH"]
ASTRA_DB_CLIENT_ID = "token"
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]

def getCqlSession():
    cluster = Cluster(
        cloud={
            "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH,
        },
        auth_provider=PlainTextAuthProvider(
            ASTRA_DB_CLIENT_ID,
            ASTRA_DB_APPLICATION_TOKEN,
        ),
    )
    astraSession = cluster.connect()
    return astraSession

def getLocalCassandraSession():
    # A session to a locally-running Cassandra
    cluster = Cluster()
    localSession = cluster.connect()
    return localSession
