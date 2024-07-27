from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np

def initDB():

    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

    utility.drop_collection('job_descriptions')

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id = False, description='id'),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768, description='embeddings')
    ]

    schema = CollectionSchema(fields)
    collection = Collection('job_descriptions', schema, consistency_level="Strong")

    embeddings = np.loadtxt('embeddings.txt', delimiter=',')
    data = [np.arange(embeddings.shape[0]), embeddings]

    for i in range(embeddings.shape[0]): insert_result = collection.insert([[data[0][i]], [data[1][i]]])
    collection.flush()

    index = {
        "index_type": "IVF_FLAT", 
        "metric_type": "COSINE", 
        "params": {"nlist": 128}
        }

    collection.create_index("embeddings", index)