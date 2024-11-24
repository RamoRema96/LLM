from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool

from tool import get_current_position, get_supermarkets_nearby, get_recipes


position_tool = FunctionTool.from_defaults(fn=get_current_position)
supermarket_tool = FunctionTool.from_defaults(fn=get_supermarkets_nearby)
recepies = FunctionTool.from_defaults(fn=get_recipes)

llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
agent = ReActAgent.from_tools([position_tool, supermarket_tool, recepies], llm=llm, verbose=True)
response = agent.chat("First of all advice me a recepie a can do at launch. Then obtain my actual position and then advice me all the supermarkets nearby me where I can buy the ingredient of the recepies. Once yo have all these information write the recepie, the nutritional values and the supermarket. Then stop")

print(response)


