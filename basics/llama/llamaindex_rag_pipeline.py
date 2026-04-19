import os
from dotenv import load_dotenv

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore


DATA_DIR = "basics/data/alfred_household"
CHROMA_DB_DIR = "basics/chroma_db"
CHROMA_COLLECTION_NAME = "alfred_household"


def load_documents():
    reader = SimpleDirectoryReader(input_dir=DATA_DIR)
    documents = reader.load_data()

    if not documents:
        raise ValueError(
            f"Nenhum documento foi carregado da pasta: {DATA_DIR}"
        )

    return documents


def build_vector_store():
    db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def build_embedding_model():
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )


def build_llm(hf_token: str):
    return HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.2,
        max_tokens=256,
        token=hf_token,
        provider="auto",
    )


def ingest_documents(documents, vector_store):
    embed_model = build_embedding_model()

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=120, chunk_overlap=20),
            embed_model,
        ],
        vector_store=vector_store,
    )

    nodes = pipeline.run(documents=documents)

    print("\n=== INGESTÃO CONCLUÍDA ===")
    print(f"Quantidade de documentos carregados: {len(documents)}")
    print(f"Quantidade de nodes gerados: {len(nodes)}")

    return embed_model


def build_index(vector_store, embed_model):
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index


def run_query(index, llm):
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )

    question = (
        "Plan a dinner party menu and preparation approach based only on the notes. "
        "Include dietary restrictions, budget, shopping timing and recipe simplicity."
    )

    response = query_engine.query(question)

    print("\n=== PERGUNTA ===")
    print(question)

    print("\n=== RESPOSTA DO QUERY ENGINE ===")
    print(str(response))

    print("\n=== TRECHOS RECUPERADOS ===")
    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\n--- Source Node {i} | score={node.score} ---")
        print(node.text)

    return query_engine, response


def evaluate_response(query_engine, llm):
    evaluator = FaithfulnessEvaluator(llm=llm)

    eval_question = (
        "What dietary restrictions should Alfred consider for the dinner party?"
    )
    eval_response = query_engine.query(eval_question)
    eval_result = evaluator.evaluate_response(response=eval_response)

    print("\n=== AVALIAÇÃO DE FAITHFULNESS ===")
    print(f"Pergunta avaliada: {eval_question}")
    print(f"Resposta: {str(eval_response)}")
    print(f"Passing: {eval_result.passing}")
    print(f"Feedback: {eval_result.feedback}")


def main():
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN não encontrado. Verifique o arquivo .env."
        )

    print("HF_TOKEN carregado com sucesso.")

    documents = load_documents()
    vector_store = build_vector_store()
    embed_model = ingest_documents(documents, vector_store)
    index = build_index(vector_store, embed_model)
    llm = build_llm(hf_token)

    query_engine, _ = run_query(index, llm)
    evaluate_response(query_engine, llm)


if __name__ == "__main__":
    main()