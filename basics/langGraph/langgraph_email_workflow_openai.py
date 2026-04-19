import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def build_model() -> ChatOpenAI:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não encontrado no .env.")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key,
    )