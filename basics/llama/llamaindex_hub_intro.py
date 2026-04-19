import os
from dotenv import load_dotenv

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main() -> None:
    # Carrega variáveis do arquivo .env
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN não foi encontrado no ambiente. "
            "Verifique seu arquivo .env ou variáveis de ambiente."
        )

    print("HF_TOKEN carregado com sucesso.")

    # LLM remoto via Hugging Face Inference API
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.7,
        max_tokens=100,
        token=hf_token,
        provider="auto",
    )

    prompt = "Hello, how are you?"
    response = llm.complete(prompt)

    print("\n=== TESTE DO LLM ===")
    print(f"Prompt: {prompt}")
    print("Resposta:")
    print(str(response))

    # Embedding model
    # Aqui estamos usando um embedding model Hugging Face pelo pacote
    # de embeddings do LlamaIndex.
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    text = "AI agents can reason, act and observe."
    embedding = embed_model.get_text_embedding(text)

    print("\n=== TESTE DE EMBEDDING ===")
    print(f"Texto: {text}")
    print(f"Tamanho do vetor: {len(embedding)}")
    print(f"Primeiros 5 valores: {embedding[:5]}")


if __name__ == "__main__":
    main()