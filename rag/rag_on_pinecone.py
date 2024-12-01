

from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os
import pandas as pd
from tqdm import tqdm 
from pinecone.grpc import PineconeGRPC as Pinecone


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


load_dotenv(".localenv")
PINECONE_KEY = os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=PINECONE_KEY)

def retrieve_documents(
    namespace: str,
    query: str,
    top_k: int = 1,
    metadata_filters=None,
) -> pd.DataFrame:
    """
    Retrieves documents from Pinecone, filters out undesired keys, and returns a DataFrame.

    :param namespace: The namespace to search within.
    :param query: The query to search for.
    :param top_k: The number of top documents to retrieve.
    :param keys_to_drop: List of keys to drop from the document metadata.
    :param metadata_filters: Filters to apply to the metadata.
    """
    # Initialize Pinecone index and query embeddings
    index = pc.Index("llama-hackathon-256")

    # Perform vector search using the embedded query
    client = OpenAI()
    # Create embeddings using the client
    response = client.embeddings.create(
        model="text-embedding-3-small", input=query, encoding_format="float"
    )
    
    # Cut and normalize the embedding
    cut_dim = response.data[0].embedding[:256]
    embedded_query = normalize_l2(cut_dim)
    search_results = index.query(
        vector=embedded_query,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=metadata_filters,
    )
    results = search_results.matches[0]["metadata"]["description"]
    return results

results = retrieve_documents(namespace="recipes", query="Mi consigli qualcosa a base di pesce")

# print(type(results))
print(results)
#print(results.matches[0]["metadata"]["description"])
