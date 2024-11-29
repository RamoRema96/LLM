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
    lambda row: {"recipe_id": row["recipe_id"], "title": row["title"]}, axis=1
)

print(chunked_recipes)

# Prepare documents for embedding
docs = [{"text": row["description"], "metadata": row["metadata"]} for _, row in chunked_recipes.iterrows()]

# # Initialize the vector store
vectordb = VectorStore()

# Load or create the db
vectorStore = vectordb.load_or_create_db(docs=docs)