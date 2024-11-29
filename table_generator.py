import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta

# Database and table setup
DB_NAME = "local.db"
TABLE_NAME = "weekly_meals"

def generate_mock_data():
    """Generates 300 random records for weekly meals."""
    # Define random data generation ranges
    start_date = datetime.now()
    meal_types = ["Breakfast", "Lunch", "Dinner", "Snack"]
    
    data = []
    for i in range(300):
        date = (start_date + timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        food_id = random.randint(1, 50)  # Random food IDs
        recipe_id = random.randint(1, 20)  # Random recipe IDs
        meal_type = random.choice(meal_types)
        data.append({"date": date, "foodID": food_id, "recipeID": recipe_id, "mealType": meal_type})
    
    return pd.DataFrame(data)

def create_table_and_insert_data(df, db_name, table_name):
    """Creates a table in SQLite and inserts the data."""
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        # Create the table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                foodID INTEGER NOT NULL,
                recipeID INTEGER NOT NULL,
                mealType TEXT NOT NULL
            )
        """)
        conn.commit()
        
        # Insert the data
        df.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"Inserted {len(df)} records into the {table_name} table.")

if __name__ == "__main__":
    # Generate mock data
    df = generate_mock_data()
    
    # Insert the data into the SQLite database
    create_table_and_insert_data(df, DB_NAME, TABLE_NAME)
    
    print(f"Data successfully saved to {DB_NAME} in table {TABLE_NAME}.")
