from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os
import pandas as pd
from tqdm import tqdm 

load_dotenv(".localenv")

client = OpenAI()

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

path_data = os.path.join(os.path.dirname(__file__), "data", "recipes_table.csv")
recipes_df = pd.read_csv(path_data)


# Initialize an empty list to store the embeddings
embeddings_list = []

# Iterate over rows and process embeddings
for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df)):
    description = row["description"]
    
    # Create embeddings using the client
    response = client.embeddings.create(
        model="text-embedding-3-small", input=description, encoding_format="float"
    )
    
    # Cut and normalize the embedding
    cut_dim = response.data[0].embedding[:256]
    norm_dim = normalize_l2(cut_dim)
    
    # Append the normalized embedding to the list
    embeddings_list.append(norm_dim)

# Add the embeddings as a new column to the DataFrame
recipes_df["embeddings"] = embeddings_list

# Optionally save the updated DataFrame to a new CSV file
recipes_df.to_csv("recipes.csv", index=False)

print(f"Updated DataFrame saved to recipes.csv")

    