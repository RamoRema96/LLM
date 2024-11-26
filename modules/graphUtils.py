import json
from langchain_core.messages import ToolMessage
from typing import  List, Dict
from langchain.agents import tool
import os 
from fatsecret import Fatsecret
import numpy as np
from scipy.optimize import minimize
from dotenv import load_dotenv

load_dotenv()

CONSUMER_KEY_FATSECRET = os.getenv("CONSUMER_KEY_FATSECRET")
CONSUMER_SECRET_FATSECRET = os.getenv("CONSUMER_SECRET_FATSECRET")
print(CONSUMER_KEY_FATSECRET, CONSUMER_SECRET_FATSECRET)
fs = Fatsecret(CONSUMER_KEY_FATSECRET, CONSUMER_SECRET_FATSECRET)

# Tool node definition
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
        for tool_call in message.tool_calls:
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
    
# Tool definitions
@tool
def quantity_optimizer(selected_food):
    """
    Optimizes the quantities of selected foods to meet specific nutritional and caloric constraints.

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
    K_MAX = MEAL_WEIGHT * 2000           # Maximum calories (kcal)
    K_MIN = MEAL_WEIGHT * 1800           # Minimum calories (kcal)

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

