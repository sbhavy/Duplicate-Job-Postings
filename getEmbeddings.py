import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def getEmbeddings():

  data = pd.read_csv('jobpostings.csv')
  descriptions = data['Job Description'].tolist()
  descriptions = [str(d) for d in descriptions]

  model_name = 'all-mpnet-base-v2'
  model = SentenceTransformer(model_name)

  batch_size = 4096  # Adjust batch size as needed for memory usage
  embeddings = []
  for i in range(0, len(descriptions), batch_size):
    batch = None
    try: batch = descriptions[i:i+batch_size]
    except: batch = descriptions[i:]
    encoded_sentences = model.encode(batch)
    embeddings.extend(encoded_sentences.tolist())

  np.savetxt('embeddings.txt', np.array(embeddings), delimiter = ',')