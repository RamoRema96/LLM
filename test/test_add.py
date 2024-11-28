from langchain_community.vectorstores.faiss import FAISS
from tqdm import tqdm
from langchain_community.embeddings import OllamaEmbeddings
import os

def combine_vector_stores(save_path: str, embeddings):
    """
    Combine individual FAISS indexes into a single index using merge_from.
    """
    
        # Get all saved indexes
    index_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.startswith("doc_")]

    if not index_files:
        raise ValueError("No individual vector stores found to merge.")

        # Load the first index to initialize
    combined_store = FAISS.load_local(index_files[0], embeddings=embeddings, allow_dangerous_deserialization=True)


    # Merge remaining indexes into the combined store
    for index_file in index_files[1:]:
        current_store = FAISS.load_local(
            index_file,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        combined_store.merge_from(current_store)

    
    combined_store.save_local("./combined_store")






save_path = "./faiss_index_recipes"
embedding_model="llama3.1:latest"

embedding = OllamaEmbeddings(model=embedding_model)
combine_vector_stores(save_path, embeddings=embedding)