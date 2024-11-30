
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
import pandas as pd
from typing import List
import ast

def from_df_to_pinecone(
    df: pd.DataFrame, 
    index_name: str, 
    namespace: str, 
    batch_size: int = 10
) -> List[str]:
    """
    Load data from a DataFrame into a Pinecone index in batches.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be loaded.
        index_name (str): The name of the Pinecone index.
        namespace (str): The namespace for the Pinecone index.
        batch_size (int): The number of records to load in each batch. Default is 150.

    Returns:
        List[str]: A list of IDs of the loaded records.
    """

    # checking Dataframe format
    assert all(
        col in df.columns for col in ["id", "embedding", "metadata"]
    ), "DataFrame must have columns 'id', 'embedding', and 'metadata'"

    ids_loaded = []
    index = pc.Index(index_name)

    for i in range(0, len(df), batch_size):
        try:
            batch = df.iloc[i : i + batch_size]
            new_documents = []
            ids_batch = batch["id"].tolist()

            # Process each document in the batch
            for _, row in batch.iterrows():
                new_documents.append(
                    {
                        "id": row["id"],
                        "values": row["embedding"],
                        "metadata": row["metadata"], #it must a python dictionary !
                    }
                )

            # Upsert the documents if there are any
            if new_documents:
                index.upsert(vectors=new_documents, namespace=namespace)
                # extend ids_loaded list
                ids_loaded.extend(ids_batch)
                print(f"Batch {i} pushed")

        except Exception as e:
            print(f"Error {e} - batch {i}")
            continue
    return ids_loaded



load_dotenv(".localenv")

PINECONE_KEY = os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=PINECONE_KEY)
new_documents = []

df_pinecone = pd.DataFrame()
df_recipes = pd.read_csv("recipes.csv")

df_pinecone["embedding"] = df_recipes["embeddings"]
df_pinecone["id"] = df_recipes["recipe_id"]

# Create metadata column
df_pinecone["metadata"] = df_recipes.apply(
    lambda row: {
        "recipe_id": row["recipe_id"],
        "description": row["description"],
        "title": row["title"],
    },
    axis=1
)


# Function to convert the string to a list of floats
def convert_to_list(value):
    return [float(x) for x in value.strip("[]").split()]

# Apply the function to the column
df_pinecone['embedding'] = df_pinecone['embedding'].apply(convert_to_list)
a = df_pinecone["embedding"].iloc[0]
print(len(a))


# Display the resulting DataFrame
ids_loaded = from_df_to_pinecone(df=df_pinecone, index_name="llama-hackathon-256", namespace="recipes")
