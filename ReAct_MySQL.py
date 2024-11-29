from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from modules.utils import llm

# Connect to the SQLite database
db = SQLDatabase.from_uri("sqlite:///local.db", include_tables=["weekly_meals"])
print("Database connected:", db)
# Create the SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


#TODO create a more detailed prompt where you mention how to link tables and their schema
prompt = hub.pull("hwchase17/react")
print(prompt)

# Initialize the agent with tools and prompt
agent = create_react_agent(llm, toolkit.get_tools(), prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit.get_tools(),verbose=True)

# Define the user query
example_query = "How many recipes did I alreadt try?"
v = agent_executor.invoke(
    {
        "input": example_query,
    }
)
print(v)
