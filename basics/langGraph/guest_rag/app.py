import os
from typing import TypedDict, Annotated

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import search_tool, weather_info_tool, hub_stats_tool, latest_news_tool
from retriever import guest_info_tool


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_alfred():
    """
    Monta o grafo do Alfred com tools + assistant.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError(
            "Defina a variável de ambiente HF_TOKEN ou HUGGINGFACEHUB_API_TOKEN antes de executar."
        )

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=hf_token,
    )

    chat = ChatHuggingFace(llm=llm, verbose=True)
    tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool, latest_news_tool]
    chat_with_tools = chat.bind_tools(tools)

    def assistant(state: AgentState):
        return {
            "messages": [chat_with_tools.invoke(state["messages"])],
        }

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()


def ask_alfred(alfred, user_message: str, previous_messages=None):
    """
    Helper para facilitar perguntas ao Alfred.
    """
    if previous_messages is None:
        messages = [HumanMessage(content=user_message)]
    else:
        messages = previous_messages + [HumanMessage(content=user_message)]

    response = alfred.invoke({"messages": messages})
    return response


if __name__ == "__main__":
    alfred = build_alfred()

    print("=== TESTE 1: GUEST INFO ===")
    response = ask_alfred(alfred, "Tell me about Lady Ada Lovelace.")
    print(response["messages"][-1].content)
    print()

    print("=== TESTE 2: HUB STATS ===")
    response = ask_alfred(
        alfred,
        "One of our guests is from Qwen. What can you tell me about their most popular model?",
    )
    print(response["messages"][-1].content)
    print()

    print("=== TESTE 3: WEATHER ===")
    response = ask_alfred(
        alfred,
        "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?",
    )
    print(response["messages"][-1].content)
    print()

    print("=== TESTE 4: MULTI-TOOL ===")
    response = ask_alfred(
        alfred,
        "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?",
    )
    print(response["messages"][-1].content)
    print()

    print("=== TESTE 5: MEMORY ===")
    response1 = ask_alfred(
        alfred,
        "Tell me about Lady Ada Lovelace. What's her background and how is she related to me?",
    )
    print("Primeira resposta:")
    print(response1["messages"][-1].content)
    print()

    response2 = ask_alfred(
        alfred,
        "What projects is she currently working on?",
        previous_messages=response1["messages"],
    )
    print("Segunda resposta com memória:")
    print(response2["messages"][-1].content)