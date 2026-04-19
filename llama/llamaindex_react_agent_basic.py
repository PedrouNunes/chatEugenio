import asyncio
import os

from dotenv import load_dotenv

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers and returns the result.
    """
    return a * b


def build_llm() -> HuggingFaceInferenceAPI:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN não encontrado no .env.")

    return HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.1,
        max_tokens=256,
        token=hf_token,
        provider="auto",
    )


async def main() -> None:
    llm = build_llm()

    multiply_tool = FunctionTool.from_defaults(
        fn=multiply,
        name="multiply_numbers",
        description="Útil para multiplicar dois números inteiros.",
    )

    agent = ReActAgent(
        name="MathAssistant",
        description="Agente que resolve multiplicações simples.",
        system_prompt=(
            "Você é um assistente útil. "
            "Quando a pergunta envolver multiplicação, use a tool disponível."
        ),
        tools=[multiply_tool],
        llm=llm,
    )

    print("\n=== EXECUÇÃO SEM MEMÓRIA ===")
    response = await agent.run(user_msg="What is 12 times 13?")
    print(str(response))

    print("\n=== EXECUÇÃO COM MEMÓRIA ===")
    ctx = Context(agent)

    response = await agent.run(
        user_msg="My name is Pedro.",
        ctx=ctx,
    )
    print(str(response))

    response = await agent.run(
        user_msg="What is my name?",
        ctx=ctx,
    )
    print(str(response))


if __name__ == "__main__":
    asyncio.run(main())