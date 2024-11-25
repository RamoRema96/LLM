import os
import json
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.agents import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

# references

# 1. groq model-> https://groq.com/introducing-llama-3-groq-tool-use-models/
# 1.b groq pricing -> https://groq.com/pricing/
# 2. langraph with memory -> https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot



# Load environment variables
load_dotenv()

# Initialize memory saver
memory = MemorySaver()

# Configuration for graph updates
config = {"configurable": {"thread_id": "1"}}


# Define state dictionary for the graph
class State(TypedDict):
    messages: Annotated[List, add_messages]
    pippo: Annotated[List, add_messages]


# Tool definitions
@tool
def author_info(query: str) -> str:
    """
    Provides information about the author if explicitly asked.
    """
    return "His name is Gianluca Boni, and he is 28 years old."


@tool
def magic_function(input: int) -> int:
    """
    The Oz Wizard Magic function. Input must be an integer.
    """
    return input + 2


# Tool initialization
my_tools = [author_info, magic_function]

# Initialize the LLM with tools
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0.0,
    max_retries=2,
    api_key=GROQ_API_KEY,
)
llm_with_tools = llm.bind_tools(my_tools)

# Graph definition
graph_builder = StateGraph(State)


# Chatbot node logic
def chatbot(state: State) -> Dict[str, List]:
    """
    Handles user interaction and integrates tool usage when required.
    """
    context_message = """
    You are an AI agent designed to give suggestions about healthy food. 
    However, you can access the `magic_function` and `author_info` tools in case users 
    explicitly ask for those.
    """
    messages_with_context = [SystemMessage(context_message)] + state["messages"]
    response_to_append = {"messages": [llm_with_tools.invoke(messages_with_context)]}
    return response_to_append


graph_builder.add_node("chatbot", chatbot)


# Streaming graph updates
def stream_graph_updates(user_input: str):
    """
    Streams graph updates based on user input.
    """
    new_message = HumanMessage(user_input)
    for event in graph.stream(
        {"messages": [new_message]}, config, stream_mode="values"
    ):
        message = event["messages"][-1]
        # Check if the message is either AIMessage or HumanMessage
        if isinstance(message, (ToolMessage)):
            continue  # Skip ToolMessage or other types you don't want to display

        else:
            message.pretty_print()


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


tool_node = BasicToolNode(tools=my_tools)
graph_builder.add_node("tools", tool_node)

# Graph conditional edges
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")  # Return to chatbot after tool execution
graph_builder.set_entry_point("chatbot")

# Compile the graph with memory saver
graph = graph_builder.compile(checkpointer=memory)

# Main loop for user interaction
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            print(f"Error: {e}")
            break
