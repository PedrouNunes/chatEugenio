import os

from smolagents import CodeAgent, InferenceClientModel

from tools import search_tool, weather_info_tool, hub_stats_tool, latest_news_tool
from retriever import build_guest_info_tool


def build_alfred() -> CodeAgent:
    """
    Monta o Alfred com todas as tools do projeto.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=hf_token,
    )

    guest_info_tool = build_guest_info_tool()

    alfred = CodeAgent(
        tools=[
            guest_info_tool,
            search_tool,
            weather_info_tool,
            hub_stats_tool,
            latest_news_tool,
        ],
        model=model,
        add_base_tools=True,
        planning_interval=3,
    )
    return alfred


if __name__ == "__main__":
    alfred = build_alfred()

    print("=== TESTE 1: GUEST INFO ===")
    response = alfred.run("Tell me about Lady Ada Lovelace.")
    print(response)
    print()

    print("=== TESTE 2: HUB STATS ===")
    response = alfred.run("One of our guests is from Qwen. What can you tell me about their most popular model?")
    print(response)
    print()

    print("=== TESTE 3: WEATHER ===")
    response = alfred.run("What's the weather like in Paris tonight? Will it be suitable for our fireworks display?")
    print(response)
    print()

    print("=== TESTE 4: LATEST NEWS ===")
    response = alfred.run("Give me the latest news about artificial intelligence.")
    print(response)
    print()

    print("=== TESTE 5: MEMORY ===")
    response1 = alfred.run("Tell me about Lady Ada Lovelace.")
    print("Primeira resposta:")
    print(response1)
    print()

    response2 = alfred.run("What projects is she currently working on?", reset=False)
    print("Segunda resposta com memória:")
    print(response2)