from typing import Annotated
from dotenv import load_dotenv
import os
import asyncio
import sqlite3
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables at the start, force reload to prevent caching
load_dotenv(override=True)

# Validate environment variables
api_base = os.getenv("OPENAI_API_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(graph: CompiledStateGraph, user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


def main():
    try:
        graph_builder = StateGraph(State)

        model = init_chat_model(
            "qwen-turbo", model_provider="openai", base_url=api_base, temperature=0
        )

        def chatbot(state: State):
            return {"messages": [model.invoke(state["messages"])]}

        graph_builder.add_node("chatbot", chatbot)

        graph_builder.add_edge(START, "chatbot")
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        memory = SqliteSaver(conn)
        graph = graph_builder.compile(checkpointer=memory)
        print("Chat initialized. Type 'quit', 'exit', or 'q' to end the conversation.")

        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                stream_graph_updates(graph, user_input)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                # fallback if input() is not available
                user_input = "What do you know about LangGraph?"
                print("User: " + user_input)
                stream_graph_updates(graph, user_input)
                break
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")


if __name__ == "__main__":
    main()
