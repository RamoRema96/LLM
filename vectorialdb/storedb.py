import os
import pandas as pd
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings


class VectorStore:
    def __init__(
        self, embedding_model="llama3.1:latest", save_path="./faiss_index_cv"
    ) -> None:
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.save_path = save_path

    def __create_db(self, docs: list):
        """
        Create a new vectorstore from the provided documents
        docs(list): It is a list of dictionary. It must contain the keys "text" and "metadata"
        """
        try:
            texts = [doc["text"] for doc in docs]
            metadatas = [doc["metadata"] for doc in docs]
            vectorStore = FAISS.from_texts(
                texts, embedding=self.embedding, metadatas=metadatas
            )
            vectorStore.save_local(self.save_path)
            return vectorStore
        except KeyError as e:
            print(f"KeyError: Missing key {e} in one or more documents")

    def load_or_create_db(self, docs=None):
        """
        Load the vector store form disk if it exists; otherwise it creates
        a new one if docs id provided

        docs(list): It is a list of dictionary
        """
        if os.path.exists(self.save_path):
            vectorStore = FAISS.load_local(
                self.save_path,
                embeddings=self.embedding,
                allow_dangerous_deserialization=True,
            )
        elif docs:
            vectorStore = self.__create_db(docs)
        else:
            raise ValueError("Documents must be provided to create a new database")
        return vectorStore

    def add_to_existing_db(self, new_docs:list, vectorStore: FAISS):
        """
        Add new documents to an existing vector store

        new_doc(list): List of dictionary. The dictionary must contain the keys "text" and "metadata"
        vectorStore(FAISS): The database where to add new documents
        """
        # Add new documents
        texts = [doc["text"] for doc in new_docs]
        metadatas = [doc["metadata"] for doc in new_docs]
        vectorStore.add_texts(texts, metadatas=metadatas)
        vectorStore.save_local(self.save_path)
        return vectorStore