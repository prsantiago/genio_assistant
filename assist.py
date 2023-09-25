import argparse
import logging
import yaml

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from datasets import load_dataset
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llm import OpenAI
from langchain.embeddings import OpenAIEmbeddings

logger = logging()

cloud_config = {
    'secure_connect_bundle': config["astradb"]["secure_bundle"]
}

auth_provider = PlainTextAuthProvider(config["astradb"]["clientId"], config["astradb"]["secret"])
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

llm = OpenAI(openai_api_key=config["openai"]["token"])
embedding = OpenAIEmbeddings(openai_api_key=config["openai"]["token"])

vector_store = Cassandra(
    embedding=embedding,
    session=session,
    keyspace=config["astradb"]["keyspace"],
    table_name="assistant_vector_store",
)

logging.info("Loading data from huggingface")
dataset = load_dataset("Biddls/Onion_News", split="train")
headlines = dataset

logging.info("Generating embeddings and storing in AstraDB")
vector_store.add_texts(headlines)

logging.info(f"Inserted {len(headlines)} headlines")

vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)

first_question = True

while True:
    if first_question:
        query_text = input("Enter your quesiton (or type 'quit' to exit): ")
        first_question = False
    else:
        query_text = input("What's your next quesiton? (or type 'quit' to exit): ")

    if query_text.lower() == "quit":
        break

    print(f"QUESTION: {query_text}")
    answer = vector_index.query(query_text, llm=llm).strip()
    print(f"QUESTION: {query_text}")

    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in vector_store.similarity_search_with_score(query_text, k=4):
        print(f"%0.4f {(score, doc.page_contect[:60])}")