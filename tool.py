import googlemaps
import os
import json
from langchain_community.llms import Ollama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from vectorialdb.storedb import VectorStore

load_dotenv(".localenv")

def rag(query_user:str):
    """
    Retrieves a suggested recipe based on the user's query by leveraging a retrieval-augmented generation (RAG) system.

    This function uses a vector store to load or create a database of recipes and employs a language model (GPT-3.5) 
    to generate a recipe suggestion based on the user's preferences. The system uses a combination of document retrieval 
    and language model-based generation to suggest the most relevant recipe. The user query is matched to relevant documents 
    in the vector store, which are then used to generate a response.

    Args:
        query_user (str): A string containing the user's query about a recipe. For example, a request like "I'd like a pasta dish".

    Returns:
        str: The generated response from the model, which includes the suggested recipe and an explanation of the choice.
        
    Example:
        >>> response = rag("vorrei un piatto di pasta")
        >>> print(response)
        "Ti consiglio di preparare un piatto di spaghetti aglio e olio, poiché è un piatto semplice e veloce con ingredienti comuni."

    The function follows these steps:
    1. Loads or creates a vector database (VectorStore) containing recipes.
    2. Uses a language model (`ChatOpenAI` with the `gpt-3.5-turbo-1106` model) to generate a response.
    3. Retrieves relevant documents from the vector store based on the user's query (with a search limit of 3).
    4. Uses the retrieved context to formulate a response, combining document retrieval with the model's generation capabilities.
    """
    vectordb = VectorStore()

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
            "input": f"{query_user}"
        }
)
    return response

def get_current_position() -> dict:
    """
    Retrieves the current geographic position using the Google Maps API.

    This function uses a geolocation request to determine the latitude and longitude 
    of the device and then performs a reverse geocoding operation to obtain the 
    formatted address.

    Returns:
        dict: A dictionary containing the following keys:
            - "lat" (float): The latitude of the current position.
            - "lng" (float): The longitude of the current position.
            - "address" (str): The formatted address corresponding to the location.
    """
    cwd = os.path.dirname(__file__)
    cred_path = os.path.join(cwd, "credentials", "credentials_google.json")
    with open(cred_path, "r") as f:
        creds = json.load(f)
        KEY_API_GOOGLE_MAPS = creds["KEY_API_GOOGLE_MAPS"]

    gmaps = googlemaps.Client(key=KEY_API_GOOGLE_MAPS)
    response = gmaps.geolocate()
    latitude = response["location"]["lat"]
    longitude = response["location"]["lng"]

    result = gmaps.reverse_geocode((latitude, longitude))
    position = {
        "lat": latitude,
        "lng": longitude,
        "address": result[0]["formatted_address"],
    }
    return position





def get_supermarkets_nearby(latitude, longitude, radius=2000):
    """

    Retrieves a list of supermarkets within a specified radius from a given latitude and longitude 
    using the Google Maps Places API.

    Args:
        latitude (float): The latitude of the reference location.
        longitude (float): The longitude of the reference location.
        radius (int, optional): The search radius in meters. Default is 2000 meters (2 km).

    Returns:
        list: A list of dictionaries, each containing information about a supermarket. 
              Each dictionary includes:
              - 'name' (str): The name of the supermarket.
              - 'lat' (float): The latitude of the supermarket.
              - 'lon' (float): The longitude of the supermarket.
              - 'distance_m' (str): The vicinity (address or description) of the supermarket.
    
    """

    cwd = os.path.dirname(__file__)
    cred_path = os.path.join(cwd, "credentials", "credentials_google.json")
    with open(cred_path, "r") as f:
        creds = json.load(f)
        KEY_API_GOOGLE_MAPS = creds["KEY_API_GOOGLE_MAPS"]
    gmaps = googlemaps.Client(key=KEY_API_GOOGLE_MAPS)

    # Request supermarkets (place type: supermarket) within a 2 km radius
    places_result = gmaps.places_nearby((latitude, longitude), radius=radius, type="supermarket")

    supermarkets = []
    for place in places_result.get('results', []):
        supermarkets.append({
            'name': place.get('name', 'Unknown'),
            'lat': place['geometry']['location']['lat'],
            'lon': place['geometry']['location']['lng'],
            'distance_m': place.get('vicinity', 'Unknown')
        })
    
    return supermarkets

def get_recipes(context, input_question):

    """
    Fetches a recipe suggestion based on the provided context and question.

    The function uses a pre-trained model to generate a recipe based on the input context 
    (e.g., meal type or dietary preference) and the input question (e.g., a query asking 
    for a recipe suggestion). The generated recipe includes the exact measurements of each 
    ingredient in grams and the total calories (kcals) of the recipe.

    Args:
        context (str): The context or background information for the recipe (e.g., type of meal, dietary restriction).
        input_question (str): The specific recipe query (e.g., asking for a good recipe for dinner, vegan options).

    Returns:
        str: The generated recipe in response to the input question, including ingredient amounts in grams 
             and the total calories for the recipe.
    """

    private_model = "llama3.1:latest"
    # Instantiate the model
    llm = Ollama(
        model=private_model,
        #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.4,
    )
    
    prompt = ChatPromptTemplate.from_template(
        """
    You are a great chef and you know all recipes in the world. You can advise anyone with famous recipes.
    For every recipe you provide, include the exact measurements of each ingredient in grams, and calculate the total calories (kcals) of the recipe.
    Context: {context}
    Question: {input}
    """
    )
    filled_prompt = prompt.format(context=context, input=input_question)
    
    # Call the model with the prompt
    response = llm.invoke(filled_prompt)
    
    return response
    


context = "A vegetarian meal for dinner"
input_question = "What is a good recipe?"
response = get_recipes(context, input_question)
print(response)