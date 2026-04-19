import os
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore


CHROMA_DB_DIR = "basics/chroma_db"
CHROMA_COLLECTION_NAME = "alfred_household"


def build_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def build_llm(hf_token: str) -> HuggingFaceInferenceAPI:
    return HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.2,
        max_tokens=256,
        token=hf_token,
        provider="auto",
    )


def build_query_engine():
    db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = build_embed_model()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN não encontrado no .env.")

    llm = build_llm(hf_token)

    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="compact",
    )
    return query_engine


def main() -> None:
    query_engine = build_query_engine()

    rag_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="alfred_household_query_tool",
        description=(
            "Útil para responder perguntas sobre os documentos da casa do Alfred, "
            "incluindo preferências de jantar, restrições alimentares, agenda e orçamento."
        ),
    )

    print("\n=== METADADOS DA QUERY ENGINE TOOL ===")
    print(f"Nome: {rag_tool.metadata.name}")
    print(f"Descrição: {rag_tool.metadata.description}")

    print("\n=== CHAMADA DIRETA DA TOOL ===")
    result = rag_tool.call(
        input=(
            "What dietary restrictions and menu constraints should Alfred consider "
            "for the dinner party?"
        )
    )

    print(result)


if __name__ == "__main__":
    main()