import json
from langchain_core.messages import ToolMessage
from typing import  List, Dict
from langchain.agents import tool
from langchain_groq import ChatGroq
import os 
from fatsecret import Fatsecret
import numpy as np
from scipy.optimize import minimize
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import logging

# Load environment variables
load_dotenv()

#  ~~~~~~~~~~~~~~~~~~~~~ LLM ~~~~~~~~~~~~~~~~~~
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL")
# Check if environment variables are loaded
if not GROQ_API_KEY or not LLAMA_MODEL:
    raise ValueError("GROQ_API_KEY and/or LLAMA_MODEL environment variables are not set.")
# Initialize the LLM
llm = ChatGroq(
    model=LLAMA_MODEL,
    temperature=0.0,
    max_retries=2,
    api_key=GROQ_API_KEY,
)


# ~~~~~~~~~~~~~ FatSecret Init ~~~~~~~~~~~~~~~~~~~~~~~~~
CONSUMER_KEY_FATSECRET = os.getenv("CONSUMER_KEY_FATSECRET")
CONSUMER_SECRET_FATSECRET = os.getenv("CONSUMER_SECRET_FATSECRET")
fs = Fatsecret(CONSUMER_KEY_FATSECRET, CONSUMER_SECRET_FATSECRET)


# ~~~~~~~~~~~~~~~~~~System Message, to update as new tools are added ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
system_message = """
                You are an advanced AI agent called RecipeHero designed to assist users with nutritional information and food optimization.
                Your functionality includes:
                1. **Providing nutritional insights**: Use the FatSecret API to fetch nutrient data for specific foods.
                2. **Optimizing food quantities**: Use the optimizer to calculate ideal proportions of foods. The dietary goals are pre-defined within the optimizer tool and should not be asked of the user.
                3. **Meal tracking and planning**: Use the `diet_explorer` tool to retrieve information from the diet table about past meals the user has eaten or planned meals saved for the future.

                Rules:
                - If the user asks for food-related information, use the tools available in the "FatSecret" node.
                - If the user provides specific foods and requests optimization, transition to the "Optimizer" node. Use the `quantity_optimizer` tool directly, assuming pre-configured dietary goals.
                - If the user asks about their past meals or planned meals, use the `diet_explorer` tool to query the diet table.
                  **IMPORTANT** The argument to pass must be message provided by the user using his natural language. 
                   DO NOT elaborate any sql query, the tool will handle it.
                       ### Example User Queries To pass to the sql agent:
                        - "What meals have I planned for dinner this week?"
                        - "How many recipes have I tried so far?"
                        - "What did I eat for breakfast last Monday?"
                - Avoid asking the user for dietary goals, as they are already specified.
                - For general questions unrelated to food, meal tracking, or optimization, stay in the "Chatbot" node.

                Always strive to provide clear, actionable, and user-friendly responses. Ensure the user's needs are met efficiently.
            """


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tool node Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BasicToolNode:
    """
    A node to execute tools requested in the last AIMessage.
    """

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict) -> Dict[str, List[ToolMessage]]:
        if not (messages := inputs.get("messages", [])):
            raise ValueError("No message found in input")

        message = messages[-1]
        outputs = []
        for i,tool_call in enumerate(message.tool_calls):
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
# ~~~~~~~~~~~~~~~~~~~~~~~ Tool definitions ~~~~~~~~~~~~~~~~~~~~~~~~~
@tool
def quantity_optimizer(selected_food:Dict):
    """
    Optimizes the quantities of selected foods.

    Args:
        selected_food (dict): A nested dictionary with the following structure:
            {
                "food_name": {
                    "proteins": float,       # Protein content per 100g
                    "fats": float,          # Fat content per 100g
                    "carbohydrates": float, # Carbohydrate content per 100g
                    "calories": float,      # Calorie content per 100g
                    "min_quantity": float   # Minimum quantity required (optional)
                },
                ...
            }

    Returns:
        dict: Optimal quantities of foods along with total macronutrient and calorie summaries.

    Raises:
        ValueError: If no feasible solution is found or if the input dictionary is invalid.
    """

    # Constants
    BODY_WEIGHT = 72
    MEAL_WEIGHT = 1
    P = MEAL_WEIGHT * BODY_WEIGHT * 1.5  # Target protein (grams)
    G = MEAL_WEIGHT * 70                 # Target fat (grams)
    C = MEAL_WEIGHT * 150                # Target carbohydrates (grams)
    K_MAX = MEAL_WEIGHT * 3000           # Maximum calories (kcal)
    K_MIN = MEAL_WEIGHT * 2500           # Minimum calories (kcal)

    def compute_macros(q, key):
        return sum(q[i] * selected_food[food][key] / 100 for i, food in enumerate(selected_food))

    def objective_function(q):
        total_proteins = compute_macros(q, "proteins")
        total_fats = compute_macros(q, "fats")
        total_carbohydrates = compute_macros(q, "carbohydrates")

        penalties = 0
        penalties += (P - total_proteins)**2 if total_proteins < P else (total_proteins - P) * 0.2
        penalties += (total_fats - G)**2 if total_fats > G else 0
        penalties += (total_carbohydrates - C)**2 if total_carbohydrates > C else 0
        penalties += sum((q[i] - selected_food[food]["min_quantity"])**2
                         for i, food in enumerate(selected_food)
                         if q[i] < selected_food[food]["min_quantity"])
        return penalties

    def calorie_constraint_upper(q):
        return K_MAX - compute_macros(q, "calories")

    def calorie_constraint_lower(q):
        return compute_macros(q, "calories") - K_MIN

    # Initial guesses and constraints
    initial_quantities = np.full(len(selected_food), 20)
    constraints = [
        {'type': 'ineq', 'fun': calorie_constraint_upper},
        {'type': 'ineq', 'fun': calorie_constraint_lower},
    ]
    bounds = [(0, None) for _ in selected_food]

    # Optimization
    result = minimize(objective_function, initial_quantities, constraints=constraints, bounds=bounds)
    if not result.success:
        raise ValueError("Optimization did not find a feasible solution.")

    # Generate result output
    outcome = {}
    for i, food in enumerate(selected_food):
        quantity = result.x[i]
        food_data = selected_food[food]
        outcome[food] = {
            "quantity (g)": round(quantity, 2),
            "calories (kcal)": round(quantity * food_data["calories"] / 100, 2),
            "proteins (g)": round(quantity * food_data["proteins"] / 100, 2),
            "fats (g)": round(quantity * food_data["fats"] / 100, 2),
            "carbohydrates (g)": round(quantity * food_data["carbohydrates"] / 100, 2),
        }

    # Summary
    total_proteins = compute_macros(result.x, "proteins")
    total_fats = compute_macros(result.x, "fats")
    total_carbohydrates = compute_macros(result.x, "carbohydrates")
    total_calories = compute_macros(result.x, "calories")
    outcome["summary"] = {
        "total_proteins (g)": round(total_proteins, 2),
        "total_fats (g)": round(total_fats, 2),
        "total_carbohydrates (g)": round(total_carbohydrates, 2),
        "total_calories (kcal)": round(total_calories, 2),
    }

    return outcome

@tool
def get_nutrients(food:str) -> Dict:
    """
    Retrieves the nutritional information for a specific food item using the FatSecret API.

    This function searches for a food item by name, retrieves its unique identifier from the FatSecret API, 
    and fetches its nutrient details. It specifically attempts to find the serving size that corresponds 
    to 100 grams for standardization. If no such serving is found, it returns the first available serving.

    Args:
        food (str): The name of the food item to search for.

    Returns:
        Dict: A dictionary containing the nutrient information for the selected serving of the food item.
              The keys in the dictionary include nutrient details such as calories, protein, fat, 
              carbohydrates, etc., as provided by the FatSecret API.

    Raises:
        ValueError: If the food name is empty or no matches are found.
        APIError: If there is an issue with the FatSecret API request or response.

    Example:
        >>> get_nutrients("Pasta")
            {'calcium': '1',
            'calories': '157',
            'carbohydrate': '30.68',
            'cholesterol': '0',
            'fat': '0.92',
            'fiber': '1.8',
            'iron': '7',
            'measurement_description': 'g',
            'metric_serving_amount': '100.000',
            'metric_serving_unit': 'g',
            'monounsaturated_fat': '0.130',
            'number_of_units': '100.000',
            'polyunsaturated_fat': '0.317',
            'potassium': '45',
            'protein': '5.77',
            'saturated_fat': '0.175',
            'serving_description': '100 g',
            'serving_id': '320989',
            'serving_url': 'https://www.fatsecret.com/calories-nutrition/generic/penne-cooked?portionid=320989&portionamount=100.000',
            'sodium': '233',
            'sugar': '0.56',
            'vitamin_a': '0',
            'vitamin_c': '0'
        }
    """
        
    list_food = fs.foods_search(search_expression=food)
    #get id first match: TODO have LLM that gets ID best match
    food_id = list_food[0]['food_id']
    servings=fs.food_get(food_id)['servings']['serving']
    for serving in servings:
        if serving['measurement_description'] == 'g': #and int(serving['metric_serving_amount']) == 100:
            return serving
    return servings[0]

@tool
def diet_explorer(question: str):
    """
    A tool to explore the diet table based to reply the question from the user.

    Args:
        question (str): The question asked by the user the Agent has to convert into a valid MySQL query.

    `diet` table schema:
        - day (DATE)
        - recipeID (STRING)
        - foodID (STRING)
        - quantity (FLOAT)
        - measurement (STRING)
        - typeMeal (STRING)
        - userID (STRING)

    """
    # Connect to the database and include the "weekly_meals" table
    TABLE = os.getenv("DIET_TABLE")
    db = SQLDatabase.from_uri("sqlite:///local.db", include_tables=[TABLE])


    # Replace with your actual schema
    schema = """
        - day (DATE)
        - recipeID (STRING)
        - foodID (STRING)
        - quantity (FLOAT)
        - measurement (STRING)
        - typeMeal (STRING)
        - userID (STRING)
    """

    # Define the prompt string
    prompt_string = """
    You are an advanced AI agent specialized in answering queries related to dietary habits, using the schema of a table named `diet` to fetch accurate information. 

    You don't need to fetch the diet table schema, as that is specified here:
    ### `diet` Table Schema:

        - day (DATE)
        - recipeID (STRING)
        - foodID (STRING)
        - quantity (FLOAT)
        - measurement (STRING)
        - typeMeal (STRING)
        - userID (STRING)

    You have access to the following tools:
    {tools}

    Use the following format:
    Question: the input question you must answer

    Loop thorugh the following though process until you get the correct answer.

        - Thought: you should always think about what to do
        - Action: the action to take, should be one of [{tool_names}]
        - Action Input: the input to the action
        - Observation: the result of the action

    As you get the correct answer returns:
        - Final Answer: the output from the final sql query.


    Question: {input}

    Thought:{agent_scratchpad}
    """

    # Create a PromptTemplate
    prompt = PromptTemplate(input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'], template=prompt_string)
    print(prompt)

    # Create the SQLDatabaseToolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Initialize the agent with tools and the detailed prompt
    agent = create_react_agent(llm, toolkit.get_tools(), prompt)
    agent_executor = AgentExecutor(agent=agent, 
                                   tools=toolkit.get_tools(), 
                                   max_execution_time=5,
                                   verbose=True,
                                   handle_parsing_errors=True,
                                   return_intermediate_steps=False
                                   )

    # Invoke the agent with the provided query
    try:
        response = agent_executor.invoke({"input": question})
        return response

    except Exception as e:
        logging.error(f"Error during agent execution: {e}")
        raise

