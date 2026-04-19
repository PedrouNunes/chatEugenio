import asyncio
import os

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from tools import search_tool, weather_info_tool, hub_stats_tool, latest_news_tool
from retriever import guest_info_tool


def build_alfred() -> AgentWorkflow:
    """
    Monta o Alfred usando AgentWorkflow.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=hf_token,
    )

    alfred = AgentWorkflow.from_tools_or_functions(
        [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool, latest_news_tool],
        llm=llm,
    )
    return alfred


async def main():
    alfred = build_alfred()

    print("=== TESTE 1: GUEST INFO ===")
    response = await alfred.run(user_msg="Tell me about Lady Ada Lovelace.")
    print(response)
    print()

    print("=== TESTE 2: HUB STATS ===")
    response = await alfred.run(
        user_msg="One of our guests is from Qwen. What can you tell me about their most popular model?"
    )
    print(response)
    print()

    print("=== TESTE 3: WEATHER ===")
    response = await alfred.run(
        user_msg="What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
    )
    print(response)
    print()

    print("=== TESTE 4: MULTI-TOOL ===")
    response = await alfred.run(
        user_msg="I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
    )
    print(response)
    print()

    print("=== TESTE 5: MEMORY ===")
    ctx = Context(alfred)

    response1 = await alfred.run(
        user_msg="Tell me about Lady Ada Lovelace.",
        ctx=ctx,
    )
    print("Primeira resposta:")
    print(response1)
    print()

    response2 = await alfred.run(
        user_msg="What projects is she currently working on?",
        ctx=ctx,
    )
    print("Segunda resposta com memória:")
    print(response2)


if __name__ == "__main__":
    asyncio.run(main())