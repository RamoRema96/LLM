import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
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

chunked_recipes = df.apply(chunk_recipe, axis=1).explode().reset_index(drop=True)


# Prepare documents for embedding
docs = [{"text": chunk["text"], "metadata": chunk["metadata"]} for chunk in chunked_recipes]

# Initialize the vector store
private_model = "llama3.1:latest"
vectordb = VectorStore(embedding_model=private_model, save_path="./faiss_index_recipes")


# Load or create the db
vectorStore = vectordb.load_or_create_db(docs=docs)