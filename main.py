from pymilvus import Collection, connections, utility
import numpy as np
from sentence_transformers import SentenceTransformer
from initDB import initDB
from getEmbeddings import getEmbeddings
import os

if not os.path.isfile('embeddings.txt'): getEmbeddings()

try:
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus.")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
    raise

collection_name = "job_descriptions"

if not utility.has_collection: initDB()

collection = Collection(name=collection_name)
collection.load()

embeddings = np.loadtxt('embeddings.txt', delimiter=',')

def search_and_query(collection, search_vectors, search_field, search_params):
    result = collection.search(search_vectors, search_field, search_params, limit=3, output_fields=["id"])
    return result

model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

query_vector = model.encode("Sample job description here")
print(search_and_query(collection, [query_vector], "embeddings", {"metric_type": "COSINE", "params": {"nprobe": 10}}))