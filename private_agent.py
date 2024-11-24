from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from tool import get_current_position
from tool import get_supermarkets_nearby


position_tool = FunctionTool.from_defaults(fn=get_current_position)
supermarket_tool = FunctionTool.from_defaults(fn=get_supermarkets_nearby)

llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
agent = ReActAgent.from_tools([position_tool, supermarket_tool], llm=llm, verbose=True)
response = agent.chat("Tell me what are all the supermarket nearby in a radius of 1 km respect to my actual position")

print(response)


