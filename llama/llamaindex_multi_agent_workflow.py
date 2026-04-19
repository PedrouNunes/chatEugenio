import asyncio
import os

from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore


CHROMA_DB_DIR = "basics/chroma_db"
CHROMA_COLLECTION_NAME = "alfred_household"


def add(a: float, b: float) -> float:
    """
    Add two numbers.
    """
    return a + b


def divide(a: float, b: float) -> float:
    """
    Divide two numbers.
    """
    if b == 0:
        raise ValueError("Não é possível dividir por zero.")
    return a / b


def build_llm() -> HuggingFaceInferenceAPI:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN não encontrado no .env.")

    return HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.1,
        max_tokens=300,
        token=hf_token,
        provider="auto",
    )


def build_query_engine_tool(llm: HuggingFaceInferenceAPI) -> QueryEngineTool:
    db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode="compact",
    )

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="alfred_household_query_tool",
        description=(
            "Consulta os documentos da casa do Alfred para responder perguntas "
            "sobre jantar, restrições alimentares, agenda, compras e orçamento."
        ),
        return_direct=False,
    )


async def main() -> None:
    llm = build_llm()

    add_tool = FunctionTool.from_defaults(
        fn=add,
        name="add_numbers",
        description="Útil para somar dois números.",
    )

    divide_tool = FunctionTool.from_defaults(
        fn=divide,
        name="divide_numbers",
        description="Útil para dividir dois números.",
    )

    rag_tool = build_query_engine_tool(llm)

    calculator_agent = ReActAgent(
        name="CalculatorAgent",
        description="Especialista em contas matemáticas simples.",
        system_prompt=(
            "Você é um agente especialista em matemática básica. "
            "Use suas tools para contas. "
            "Se a tarefa exigir informações dos documentos do Alfred, "
            "faça handoff para HouseholdAgent."
        ),
        tools=[add_tool, divide_tool],
        llm=llm,
        can_handoff_to=["HouseholdAgent"],
    )

    household_agent = ReActAgent(
        name="HouseholdAgent",
        description="Especialista nos documentos do Alfred e planejamento do jantar.",
        system_prompt=(
            "Você é um agente especialista em consultar os documentos da casa do Alfred. "
            "Use sua tool de consulta para responder perguntas sobre cardápio, "
            "restrições e logística. "
            "Se precisar fazer uma conta simples, faça handoff para CalculatorAgent."
        ),
        tools=[rag_tool],
        llm=llm,
        can_handoff_to=["CalculatorAgent"],
    )

    workflow = AgentWorkflow(
        agents=[calculator_agent, household_agent],
        root_agent=household_agent.name,
    )

    user_msg = (
        "Plan a dinner party using the household notes, then compute the budget per "
        "person for 6 guests with a total budget of 60 euros."
    )

    response = await workflow.run(user_msg=user_msg)

    print("\n=== PERGUNTA ===")
    print(user_msg)

    print("\n=== RESPOSTA DO MULTI-AGENT ===")
    print(str(response))


if __name__ == "__main__":
    asyncio.run(main())