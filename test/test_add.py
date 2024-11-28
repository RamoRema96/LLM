import os 

from langchain_community.vectorstores.faiss import FAISS
from tqdm import tqdm
from langchain_community.embeddings import OllamaEmbeddings

def combine_vector_stores(save_path:str, embeddings):
    """
    Combine individual FAISS indexes into a single index using merge_from.
    """
    try:
        # Get all saved indexes
        index_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.startswith("doc_")]

        if not index_files:
            raise ValueError("No individual vector stores found to merge.")

        # Load the first index to initialize
        combined_store = FAISS.load_local(index_files[0], embeddings=embedding, allow_dangerous_deserialization=True)

        # Merge all subsequent indexes into the combined index
        for index_file in tqdm(index_files[1:], desc="Merging vector stores"):
            new_store = FAISS.load_local(index_file, embeddings, allow_dangerous_deserialization=True)
            combined_store.index.merge_from(new_store.index)

        # Save the combined vector store
        combined_store.save_local(os.path.join(save_path, "combined_store"))

        print("Combined vector store saved successfully.")
        return combined_store

    except Exception as e:
        print(f"Error while merging vector stores: {e}")

save_path = "./faiss_index_recipes"
embedding_model="llama3.1:latest"

embedding = OllamaEmbeddings(model=embedding_model)
combine_vector_stores(save_path, embeddings=embedding)

        
