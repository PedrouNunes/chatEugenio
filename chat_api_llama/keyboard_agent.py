# chat_api_llama/keyboard_agent.py
from smolagents import LiteLLMModel, tool

# ── Trocar aqui para mudar o modelo ──
OLLAMA_MODEL = "ollama/qwen2.5-coder:3b"
# "ollama/llama3.2:3b"   → mais leve, boa qualidade geral
# "ollama/llama3.1:8b"   → mais potente, mais lento

_MODEL = LiteLLMModel(
    model_id=OLLAMA_MODEL,
    api_base="http://localhost:11434",
    max_tokens=1000,
    temperature=0.1,
)

SYSTEM_PROMPT = """
You are an expert assistant that creates keyboard files (.tec) for the Eugénio
Augmentative and Alternative Communication (AAC) system.

Generate a valid .tec file based on the user's description.
Return ONLY the file content — no explanation, no markdown, no code blocks.

════════════════════════════════════════
STRUCTURE RULES
════════════════════════════════════════

1. Every section starts with LINHA followed by GRUPO with the same (or related) name.
2. Spaces in names are replaced by ;;; (e.g. "números e backspace" → "números;;;e;;;backspace")
3. First key of EVERY group is always: TECLA TECLA_VAZIA
4. Regular letter/number keys:
   TECLA TECLA_NORMAL [letter] [label] [value] 1 -1 -1
   Example: TECLA TECLA_NORMAL A a a 1 -1 -1
   Example: TECLA TECLA_NORMAL água água água 1 -1 -1
   Example: TECLA TECLA_NORMAL pão pão pão 1 -1 -1
5. Prediction keys: TECLA TECLA_PREDICAO  /  TECLA TECLA_PREDICAO_FRASE

════════════════════════════════════════
ACTION BUTTONS — copy these EXACTLY, character by character
════════════════════════════════════════

KEYBOARD SHORTCUTS (value = key name directly):
  TECLA TECLA_NORMAL <---> apagar BkSp 1 -1 -1
  TECLA TECLA_NORMAL <---> espaço Space 1 -1 -1
  TECLA TECLA_NORMAL <---> enter Enter 1 -1 -1
  TECLA TECLA_NORMAL <---> tabulação Tab 1 -1 -1
  TECLA TECLA_NORMAL <---> capslock CapsLock 1 -1 -1
  TECLA TECLA_NORMAL <---> Shift Shift 1 -1 -1
  TECLA TECLA_NORMAL <---> Up Up 1 -1 -1
  TECLA TECLA_NORMAL <---> Down Down 1 -1 -1
  TECLA TECLA_NORMAL <---> Left Left 1 -1 -1
  TECLA TECLA_NORMAL <---> Right Right 1 -1 -1

EUGÉNIO SYSTEM ACTIONS (value = [Action-Name] with brackets):
  TECLA TECLA_NORMAL ... Mais;;;predições;;;de;;;vocabulário [Show-Next-Predictions] 1 -1 -1
  TECLA TECLA_NORMAL <---> Limpar;;;editor;;;de;;;mensagens [Clear-Editing-Area] 1 -1 -1
  TECLA TECLA_NORMAL <---> Escrever;;;em;;;aplicação;;;externa [Send-Message-To-External-Editor] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;palavra [Synthesize-Word-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;frase [Synthesize-Sentence-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;todo;;;o;;;texto [Synthesize-All-Text-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Desligar [Quit-Application] 1 -1 -1

════════════════════════════════════════
COMPLETE EXAMPLE (real file — study this carefully)
════════════════════════════════════════

LINHA números;;;e;;;backspace
GRUPO números;;;e;;;backspace
TECLA TECLA_VAZIA
TECLA TECLA_NORMAL 1 um 1 1 -1 -1
TECLA TECLA_NORMAL 2 dois 2 1 -1 -1
TECLA TECLA_NORMAL 3 três 3 1 -1 -1
TECLA TECLA_NORMAL 0 zero 0 1 -1 -1
TECLA TECLA_NORMAL <---> apagar BkSp 1 -1 -1

LINHA letras;;;e;;;ações
GRUPO letras;;;e;;;ações
TECLA TECLA_VAZIA
TECLA TECLA_NORMAL <---> Shift Shift 1 -1 -1
TECLA TECLA_NORMAL A a a 1 -1 -1
TECLA TECLA_NORMAL B b b 1 -1 -1
TECLA TECLA_NORMAL C c c 1 -1 -1
TECLA TECLA_NORMAL <---> enter Enter 1 -1 -1
TECLA TECLA_NORMAL <---> espaço Space 1 -1 -1
TECLA TECLA_NORMAL <---> Limpar;;;editor;;;de;;;mensagens [Clear-Editing-Area] 1 -1 -1

LINHA síntese;;;de;;;fala
GRUPO síntese;;;de;;;fala
TECLA TECLA_VAZIA
TECLA TECLA_NORMAL <---> Sintetizar;;;palavra [Synthesize-Word-To-Speech] 1 -1 -1
TECLA TECLA_NORMAL <---> Sintetizar;;;frase [Synthesize-Sentence-To-Speech] 1 -1 -1
TECLA TECLA_NORMAL <---> Sintetizar;;;todo;;;o;;;texto [Synthesize-All-Text-To-Speech] 1 -1 -1
TECLA TECLA_NORMAL <---> Desligar [Quit-Application] 1 -1 -1

════════════════════════════════════════
CRITICAL REMINDERS
════════════════════════════════════════
- Action buttons ALWAYS use <---> as the image field
- BkSp not Backspace, Space not Espaco, CapsLock not CAPSLOCK
- [Synthesize-Sentence-To-Speech] not [Synthesize-Phrase-To-Speech]
- Words with accents (água, pão, leite, café) go directly as text — do NOT escape them
- Return ONLY the .tec content, nothing else
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
    from smolagents.models import ChatMessage

    messages = [
        ChatMessage(role="system", content=[{"type": "text", "text": SYSTEM_PROMPT}]),
        ChatMessage(role="user",   content=[{"type": "text", "text": description}]),
    ]

    response = _MODEL(messages)

    if isinstance(response.content, list):
        return response.content[0].get("text", "")
    return response.content