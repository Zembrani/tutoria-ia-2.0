
# Adiciona destaque e cor ao chat usando Rich
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

# langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

from datetime import datetime
from dotenv import load_dotenv
import os
# OpenAi SDK
from openai import OpenAI
load_dotenv()

class PureOpenAi:
		def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
			self.api_key = api_key or os.getenv("OPENAI_API_KEY")
			self.model = model
			self.client = OpenAI(api_key=self.api_key)

		def ask(self, prompt: str) -> str:
			try:
				response = self.client.responses.create(
					model=self.model,
					input=[
						{"role": "system", "content": ""},
						{"role": "user", "content": prompt}
					]
				)
				return response.output_text.strip()
			except Exception as e:
				return f"[Erro ao chamar OpenAI: {e}]"

class ChainApp:
		def __init__(self, model: str = "gpt-4o-mini"):
			self.llm = init_chat_model(model=model)
			self.chain = self.build_chain()

		def build_chain(self):
			prompt = ChatPromptTemplate.from_messages([
				("system", ""),
				("human", "{question}")
			])
			chain = ( prompt | self.llm | StrOutputParser() )
			return chain

		def ask(self, prompt: str):
			return self.chain.invoke({ "question": prompt })

class GraphApp:
		def __init__(self, model: str = "gpt-4o-mini"):
			self.llm = init_chat_model(model=model)
			self.thread = { "configurable": { "thread_id": "fixed" }}
			self.graph = self.build_graph()

		def save_graph_schema(self, graph):
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			file_path = f"graph_{timestamp}.mmd"

			mermaid_code = graph.get_graph(xray=True).draw_mermaid()
			with open(file_path, "w") as f:
				f.write(mermaid_code)

		def make_assistant_node(self, system_prompt):
			prompt = ChatPromptTemplate.from_messages([
				("system", system_prompt),
				("human", "{messages}")
			])
			chain = ( prompt | self.llm )
			def assistant_node(state: MessagesState) -> MessagesState:
				response = chain.invoke({ "messages": state["messages"]})
				print("ENTREI NO NODO")
				return { "messages": response } 
			return assistant_node
	
		def build_graph(self):
			assistant_node = self.make_assistant_node("Voce é um assistente chamado Optimus")
			assistant_node_2 = self.make_assistant_node("Voce é um assistente chamado Bumblebee")

			builder = StateGraph(MessagesState)
			
			builder.add_node("assistant", assistant_node)
			builder.add_node("assistant_2", assistant_node_2)

			builder.add_edge(START, "assistant")
			builder.add_edge("assistant", "assistant_2")
			builder.add_edge("assistant_2", END)

			memory = MemorySaver()
			
			graph = builder.compile(checkpointer=memory)
			self.save_graph_schema(graph)
			return graph

		def ask(self, prompt: str):
			initial_state = { "messages": [ prompt ]}
			response = self.graph.invoke(input=initial_state, config=self.thread)

			return response["messages"][-1].content

llm = GraphApp()
# Classe stub para processar mensagens do chat
class ChatBot:
		def process(self, user_input: str) -> str:
			return llm.ask(user_input)

def main():
	bot = ChatBot()
	console = Console()
	console.print(Panel("Bem-vindo ao [bold cyan]Chat CLI[/bold cyan]! ([green]digite 'sair' para encerrar[/green])", style="bold magenta"))
	while True:
		user_input = Prompt.ask("[bold blue]Você[/bold blue]", console=console)
		if user_input.strip().lower() == 'sair':
			console.print("[yellow]Encerrando o chat. Até logo![/yellow]")
			break
		resposta = bot.process(user_input)
		console.print(Panel(f"[bold white]{resposta}[/bold white]", title="[bold green]Bot[/bold green]", style="green"))

if __name__ == "__main__":
	main()
