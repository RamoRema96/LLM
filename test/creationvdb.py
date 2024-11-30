import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv


load_dotenv(".localenv")

PINECONE_KEY = os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=PINECONE_KEY)

# Add the root directory of your project to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)


from vectorialdb.storedb import VectorStore

db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recipes_table.csv")

df = pd.read_csv(db_path)

# Function to chunk recipe descriptions
def chunk_recipe(recipe_row):
    """
    Splits the recipe description into chunks and associates metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = []
    for i, chunk in enumerate(text_splitter.split_text(recipe_row["description"])):
        chunks.append(
            {
                "id": f"{recipe_row['recipe_id']}_chunk{i}",  # Unique chunk ID
                "text": chunk,
                "metadata": {
                    "recipe_id": recipe_row["recipe_id"],
                    "chunk_id": i,
                },
            }
        )
    return chunks

chunked_recipes = df[["recipe_id", "description", "title"]]
chunked_recipes["metadata"] = chunked_recipes.apply(
    lambda row: {"recipe_id": row["recipe_id"], "title": row["title"], "description":row["description"]}, axis=1
)

# print(chunked_recipes)

# Prepare documents for embedding
docs = [{"text": row["description"], "metadata": row["metadata"]} for _, row in chunked_recipes.iterrows()]

# # Initialize the vector store
vectordb = VectorStore(save_path="faiss_index_recipes_384", embedding_model="paraphrase-MiniLM-L12-v2")

# Load or create the db
vectorStore = vectordb.load_or_create_db(docs=docs)


import numpy as np

# Access the underlying FAISS index
faiss_index = vectorStore.index

# Get the number of stored vectors
num_vectors = faiss_index.ntotal

# Reconstruct all vectors
vectors = np.array([faiss_index.reconstruct(i) for i in range(num_vectors)])

# # Inspect the vectors
# print("Vectors shape:", vectors.shape)
# print("Vectors:", vectors)
# Retrieve all document IDs from the docstore
doc_ids = list(vectorStore.docstore._dict.keys())

# Access metadata for each document ID
metadata = [vectorStore.docstore._dict[doc_id].metadata for doc_id in doc_ids]

index = pc.Index("llama-hackaton-384")
new_documents = []
# Combine vectors and metadata
for i, vector in enumerate(vectors):


    new_documents.append(
                    {
                        "id": doc_ids[i],
                        "values": vector,
                        "metadata": metadata[i],
                    }
                )



    index.upsert(vectors=new_documents, namespace="recipes_384")


