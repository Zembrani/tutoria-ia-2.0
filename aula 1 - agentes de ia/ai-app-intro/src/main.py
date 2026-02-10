# Adiciona destaque e cor ao chat usando Rich
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

# langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

from datetime import datetime
from dotenv import load_dotenv
import os
from typing_extensions import TypedDict, Literal
from pydantic import BaseModel, Field

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
                    {
                        "role": "system",
                        "content": ""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
            )
            return response.output_text.strip()
        except Exception as e:
            return f"[Erro ao chamar OpenAI: {e}]"


class ChainApp:

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = init_chat_model(model=model)
        self.chain = self.build_chain()

    def build_chain(self):
        prompt = ChatPromptTemplate.from_messages([("system", ""),
                                                   ("human", "{question}")])
        chain = prompt | self.llm | StrOutputParser()
        return chain

    def ask(self, prompt: str):
        return self.chain.invoke({"question": prompt})


# Schema for structured output routing
class Route(BaseModel):
    step: Literal["faroeste", "transformers"] = Field(
        None,
        description=
        "The specialist to route to: 'faroeste' for western movies or 'transformers' for Transformers movies",
    )


class State(TypedDict):
    input: str
    decision: str
    output: str


class GraphApp:

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = init_chat_model(model=model)
        self.router_llm = init_chat_model(
            model=model).with_structured_output(Route)
        self.thread = {"configurable": {"thread_id": "fixed"}}
        self.graph = self.build_graph()

    def save_graph_schema(self, graph):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"graph_{timestamp}.mmd"

        mermaid_code = graph.get_graph(xray=True).draw_mermaid()
        with open(file_path, "w") as f:
            f.write(mermaid_code)

    def make_assistant_node(self, system_prompt):
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                   ("human", "{messages}")])
        chain = prompt | self.llm

        def assistant_node(state: MessagesState) -> MessagesState:
            response = chain.invoke({"messages": state["messages"]})
            return {"messages": response}

        return assistant_node

    def padrão(self, state: MessagesState):
        """Default node - irá falar sobre qualquer filme"""
        print("Entrou no nó DEFAULT")
        response = self.llm.invoke([
            SystemMessage(
                content=
                "Você é João, você conhece todos os tipos de filmes. Sempre responda com entusiasmo sobre qualquer filme que o usuário perguntar!"
            ),
            HumanMessage(content=state["messages"][-1].content)
        ])
        return {"messages": [response]}

    def faroeste(self, state: MessagesState):
        """Django - Especialista em filmes de faroeste"""
        print("Entrou no nó FAROESTE")
        response = self.llm.invoke([
            SystemMessage(
                content=
                "Você é Django, um especialista apaixonado por filmes de faroeste. Você conhece todos os clássicos westerns, diretores como Sergio Leone, John Ford, e Clint Eastwood. Sempre responda com entusiasmo sobre cowboys, duelos e o velho oeste!"
            ),
            HumanMessage(content=state["messages"][-1].content)
        ])
        return {"messages": [response]}

    def transformer(self, state: MessagesState):
        print("Entrou no nó TRANSFORMER")
        """Bumblebee - Especialista em filmes Transformers"""
        response = self.llm.invoke([
            SystemMessage(
                content=
                "Você é Bumblebee, um especialista em filmes Transformers. Você conhece todos os filmes da franquia, personagens, Autobots, Decepticons e a história completa. Sempre responda com energia sobre Transformers!"
            ),
            HumanMessage(content=state["messages"][-1].content)
        ])
        return {"messages": [response]}

    def router(self, state: MessagesState):
        """Router que decide qual especialista deve responder"""
        decision = self.router_llm.invoke([
            SystemMessage(
                content=
                "You are a routing agent in a LangGraph workflow. Your sole task is to analyze the users input message and route it to one of three options: faroeste (western film theme), transformers (Transformers film theme with robots and action), or return __END__. Be conservative in your routing—only route to a theme if the user explicitly mentions or directly requests elements tied to it. If in doubt, always default to __END__.Follow this step-by-step process:Read the entire user message carefully.Check for explicit faroeste indicators: Look only for direct words or phrases like faroeste, western, cowboy, saloon, duel, wild west, or clear requests for a cowboy-themed story/response. Ignore vague implications (e.g., adventure or fight alone does not count).Check for explicit transformers indicators: Look only for direct words or phrases like transformers, Autobots, Decepticons, Optimus Prime, robots transforming, sci-fi robots, or clear requests for a Transformers-themed story/response. Ignore vague implications (e.g., robots or cars alone does not count).If neither theme has an explicit match, or the message is general/ambiguous/unrelated (e.g., questions about weather, math, or no theme mentioned), route to __END__.Examples:User: Tell me a story about cowboys in the wild west. → Route to faroeste-User: I want a Transformers adventure with Optimus Prime. → Route to transformers-User: Whats the weather like? → Route to __END__ (no theme match)-User: Robots are cool. → Route to __END__ (vague, not explicitly Transformers)-User: Adventure time! → Route to __END__ (ambiguous, no explicit theme)"
            ),
            HumanMessage(content=state["messages"][-1].content),
        ])
        print("Decisão do roteador:", decision.step)
        return {"decision": decision.step}

    def route_decision(self, state: State):
        """Conditional edge function - returns the node name to visit next"""
        decision = state.get("decision", "").lower()
        if "faroeste" in decision:
            return "faroeste"
        elif "transformers" in decision:
            return "transformer"
        else:
            return "__END__"

    def build_graph(self):
        builder = StateGraph(MessagesState)

        builder.add_node("router", self.router)
        builder.add_node("faroeste", self.faroeste)
        builder.add_node("transformer", self.transformer)

        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router", self.route_decision, {  # Name returned by route_decision : Name of next node to visit
        "__END__": END,
        "faroeste": "faroeste",
        "transformer": "transformer",
        },
        )
        builder.add_edge("faroeste", END)
        builder.add_edge("transformer", END)
        memory = MemorySaver()

        graph = builder.compile(checkpointer=memory)
        self.save_graph_schema(graph)
        return graph

    def ask(self, prompt: str):
        initial_state = {"messages": [prompt]}
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
    console.print(
        Panel(
            "Bem-vindo ao [bold cyan]Chat CLI[/bold cyan]! ([green]digite 'sair' para encerrar[/green])",
            style="bold magenta",
        ))
    while True:
        user_input = Prompt.ask("[bold blue]Você[/bold blue]", console=console)
        if user_input.strip().lower() == "sair":
            console.print("[yellow]Encerrando o chat. Até logo![/yellow]")
            break
        resposta = bot.process(user_input)
        console.print(
            Panel(
                f"[bold white]{resposta}[/bold white]",
                title="[bold green]Bot[/bold green]",
                style="green",
            ))


if __name__ == "__main__":
    main()
