from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool

from tool import get_current_position, get_supermarkets_nearby, get_recipes, rag


position_tool = FunctionTool.from_defaults(fn=get_current_position)
supermarket_tool = FunctionTool.from_defaults(fn=get_supermarkets_nearby)
#recepies = FunctionTool.from_defaults(fn=get_recipes)
recepies = FunctionTool.from_defaults(fn=rag)

llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
agent = ReActAgent.from_tools([position_tool, supermarket_tool, recepies], llm=llm, verbose=True)
response = agent.chat("Ciao, vorrei mangiare un piatto di agnello. Cosa puoi consigliarmi?")

print(response)


