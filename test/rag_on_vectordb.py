

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

private_model = "llama3.1:latest"

vectordb = VectorStore(embedding_model = private_model, save_path="./combined_store")

# Load or create the db
vectorStore = vectordb.load_or_create_db()

def create_chain(vectorStore):

    # Instatiate the model
    llm = Ollama(
        model=private_model,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.4,
    )
    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
    You are a professional chef and are very good at suggest recepies that match with the request of the client. Use just the reciped that you find, do not create anything by yourself
    Context: {context}
    Question: {input}
    """
    )
    # Create LLM chain
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    # se vuoi che ti funzioni devi per forza mettere quella variabile context
    retriever = vectorStore.as_retriever(search_kwargs={"k": 4})
    retriever_chain = create_retrieval_chain(retriever, chain)
    return retriever_chain

chain = create_chain(vectorStore)
response = chain.invoke(
    {
        "input": "Voglio mangiare qualcosa di agnello stasera, cosa mi consigli?"
    }
)
print(response)