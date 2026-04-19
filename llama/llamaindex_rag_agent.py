import asyncio
import os

from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore


CHROMA_DB_DIR = "basics/chroma_db"
CHROMA_COLLECTION_NAME = "alfred_household"


def calculate_budget_per_person(total_budget: float, guest_count: int) -> str:
    """
    Useful for calculating the budget per person for a dinner event.
    """
    if guest_count <= 0:
        return "A quantidade de convidados deve ser maior que zero."

    per_person = total_budget / guest_count
    return f"O orçamento por pessoa é {per_person:.2f} euros."


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
            "Útil para responder perguntas sobre os documentos da casa do Alfred, "
            "incluindo preferências de jantar, restrições alimentares, agenda, "
            "compras e orçamento. Use perguntas completas em texto."
        ),
        return_direct=False,
    )


async def main() -> None:
    llm = build_llm()

    rag_tool = build_query_engine_tool(llm)

    budget_tool = FunctionTool.from_defaults(
        fn=calculate_budget_per_person,
        name="budget_per_person_tool",
        description="Útil para calcular orçamento por pessoa em um jantar.",
    )

    agent = ReActAgent(
        name="AlfredRAGAgent",
        description="Agente que consulta os documentos do Alfred e faz cálculos simples.",
        system_prompt=(
            "Você é Alfred, um assistente doméstico útil. "
            "Quando precisar de informações sobre preferências, restrições, agenda "
            "ou orçamento, use suas tools. "
            "Baseie a resposta nos documentos disponíveis e seja objetivo."
        ),
        tools=[rag_tool, budget_tool],
        llm=llm,
    )

    question = (
        "We have 5 guests. Based on the household notes, suggest a suitable dinner "
        "plan, mention dietary restrictions, mention when shopping should happen, "
        "and estimate the budget per person if the total budget is 60 euros."
    )

    response = await agent.run(user_msg=question)

    print("\n=== PERGUNTA ===")
    print(question)

    print("\n=== RESPOSTA DO AGENTE ===")
    print(str(response))


if __name__ == "__main__":
    asyncio.run(main())