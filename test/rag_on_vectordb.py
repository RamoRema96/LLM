

import sys
import os

# Add the root directory of your project to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)


from vectorialdb.storedb import VectorStore
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv(".localenv")

vectordb = VectorStore(save_path="faiss_index_recipes_openai")

# Load or create the db
vectorStore = vectordb.load_or_create_db()

def create_chain(vectorStore):

    # Instatiate the model
    llm = ChatOpenAI(model = "gpt-3.5-turbo-1106", temperature = 0.4)
    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Sei un assistente culinario esperto e devi suggerire il miglior piatto in base alle preferenze dell'utente.
        Usa le ricette come riferimento e non inventare nulla:
        {context}
        Domanda: {input}
        Rispondi con una ricetta suggerita, spiega il motivo della scelta.
    """
    )
    # Create LLM chain
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    # se vuoi che ti funzioni devi per forza mettere quella variabile context
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})
    retriever_chain = create_retrieval_chain(retriever, chain)
    return retriever_chain

chain = create_chain(vectorStore)
response = chain.invoke(
    {
        "input": "vorre un piatto a base di pesce"
    }
)
print(response)