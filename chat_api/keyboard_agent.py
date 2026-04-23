# chat_api/keyboard_agent.py
import os
from smolagents import CodeAgent, InferenceClientModel, tool

EXAMPLE_KEYBOARD = """
LINHA vogais
GRUPO vogais
TECLA TECLA_VAZIA
TECLA TECLA_NORMAL A a a 1 -1 -1
TECLA TECLA_NORMAL E e e 1 -1 -1
TECLA TECLA_NORMAL I i i 1 -1 -1
TECLA TECLA_NORMAL O o o 1 -1 -1
TECLA TECLA_NORMAL U u u 1 -1 -1

LINHA números
GRUPO números
TECLA TECLA_VAZIA
TECLA TECLA_NORMAL 1 um 1 1 -1 -1
TECLA TECLA_NORMAL 2 dois 2 1 -1 -1
TECLA TECLA_NORMAL 3 três 3 1 -1 -1
"""

SYSTEM_PROMPT = f"""
You are an expert assistant that creates keyboard files for the Eugénio 
Augmentative and Alternative Communication (AAC) system.

When the user describes a keyboard, you MUST generate a valid .tec file.

## FORMAT RULES (follow strictly):
- Each section starts with LINHA and GRUPO with the same name
- Spaces in names are written as ;;;
- First key of every section is always: TECLA TECLA_VAZIA
- Regular keys use: TECLA TECLA_NORMAL [image] [label] [value] 1 -1 -1
  - [image]: the symbol shown (letter, number, or <---> for action keys)
  - [label]: descriptive name (spaces replaced by ;;;)
  - [value]: what is typed/sent (letter, number, or [Action-Name])
- Action keys use [Action-Name] format, e.g. [Synthesize-Word-To-Speech]
- Encoding: use \\xNN for special characters (ç=\\xe7, ã=\\xe3, etc.)

## EXAMPLE OF A VALID FILE:
{EXAMPLE_KEYBOARD}

## YOUR TASK:
1. Understand what keyboard the user wants
2. Generate the complete .tec file content
3. Return ONLY the file content — no explanation, no markdown, no code blocks
"""

@tool
def generate_keyboard_file(description: str) -> str:
    """Generate a .tec keyboard file for the Eugénio AAC system based on a natural language description.

    Args:
        description: Natural language description of the keyboard to create,
                     including what keys, sections and layout the user wants.

    Returns:
        The complete content of the .tec keyboard file.
    """
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=os.getenv("HF_TOKEN"),
        max_tokens=2048,
        temperature=0.2,
    )

    from smolagents.models import ChatMessage
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=description),
    ]

    response = model(messages)
    return response.content


def build_keyboard_agent() -> CodeAgent:
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=os.getenv("HF_TOKEN"),
        max_tokens=2048,
        temperature=0.3,
    )

    return CodeAgent(
        model=model,
        tools=[generate_keyboard_file],
        max_steps=3,
        verbosity_level=0,
    )