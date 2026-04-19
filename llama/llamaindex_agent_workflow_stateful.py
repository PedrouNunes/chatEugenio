import asyncio
import os

from dotenv import load_dotenv

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


async def add(ctx: Context, a: int, b: int) -> int:
    """
    Add two integers and increment the function call counter in workflow state.
    """
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["num_fn_calls"] += 1

    return a + b


async def multiply(ctx: Context, a: int, b: int) -> int:
    """
    Multiply two integers and increment the function call counter in workflow state.
    """
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["num_fn_calls"] += 1

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

    multiply_agent = ReActAgent(
        name="multiply_agent",
        description="Consegue multiplicar dois inteiros.",
        system_prompt=(
            "Você é um assistente útil que pode multiplicar números. "
            "Se a tarefa for de soma, faça handoff para add_agent."
        ),
        tools=[multiply],
        llm=llm,
        can_handoff_to=["add_agent"],
    )

    add_agent = ReActAgent(
        name="add_agent",
        description="Consegue somar dois inteiros.",
        system_prompt=(
            "Você é um assistente útil que pode somar números. "
            "Se a tarefa for de multiplicação, faça handoff para multiply_agent."
        ),
        tools=[add],
        llm=llm,
        can_handoff_to=["multiply_agent"],
    )

    workflow = AgentWorkflow(
        agents=[multiply_agent, add_agent],
        root_agent="multiply_agent",
        initial_state={"num_fn_calls": 0},
        state_prompt="Current state: {state}. User message: {msg}",
    )

    ctx = Context(workflow)

    response = await workflow.run(
        user_msg="Can you add 5 and 3?",
        ctx=ctx,
    )

    print("\n=== RESPOSTA DO WORKFLOW ===")
    print(str(response))

    state = await ctx.store.get("state")

    print("\n=== ESTADO FINAL ===")
    print(state)
    print(f"num_fn_calls = {state['num_fn_calls']}")


if __name__ == "__main__":
    asyncio.run(main())