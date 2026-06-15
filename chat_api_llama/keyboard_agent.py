# chat_api_llama/keyboard_agent.py
from smolagents import LiteLLMModel, tool

OLLAMA_MODEL = "ollama/qwen2.5-coder:3b"

# Modelo instanciado uma vez ao iniciar o servidor
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
5. Prediction keys: TECLA TECLA_PREDICAO  /  TECLA TECLA_PREDICAO_FRASE

════════════════════════════════════════
ACTION BUTTONS — copy these EXACTLY
════════════════════════════════════════

KEYBOARD SHORTCUTS:
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

EUGÉNIO SYSTEM ACTIONS:
  TECLA TECLA_NORMAL ... Mais;;;predições;;;de;;;vocabulário [Show-Next-Predictions] 1 -1 -1
  TECLA TECLA_NORMAL <---> Limpar;;;editor;;;de;;;mensagens [Clear-Editing-Area] 1 -1 -1
  TECLA TECLA_NORMAL <---> Escrever;;;em;;;aplicação;;;externa [Send-Message-To-External-Editor] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;palavra [Synthesize-Word-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;frase [Synthesize-Sentence-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;todo;;;o;;;texto [Synthesize-All-Text-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Desligar [Quit-Application] 1 -1 -1

════════════════════════════════════════
COMPLETE EXAMPLE (real file)
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
- Write accented characters DIRECTLY — NEVER use escape sequences
- CORRECT:   TECLA TECLA_NORMAL ç ç ç 1 -1 -1
- CORRECT:   TECLA TECLA_NORMAL Ç ç ç 1 -1 -1
- CORRECT:   TECLA TECLA_NORMAL ã ã ã 1 -1 -1
- CORRECT:   TECLA TECLA_NORMAL água água água 1 -1 -1
- WRONG:     TECLA TECLA_NORMAL \xe7 \xe7 \xe7 1 -1 -1
- WRONG:     TECLA TECLA_NORMAL \xc3\xa7 ... 1 -1 -1
- Generate ONLY keys explicitly mentioned. Do NOT add Shift, Power, Synthesize,
  Quit, Clear, Send or any key not in the description. No exceptions.
- Return ONLY the .tec content, nothing else
"""


@tool
def generate_keyboard_file(description: str) -> str:
    """Generate a .tec keyboard file for the Eugénio AAC system.

    Args:
        description: Natural language description of the keyboard to create.

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


def generate_keyboard_with_history(description: str, history: list) -> str:
    """Gera teclado com memória de conversação anterior.

    Args:
        description: Pedido atual do utilizador.
        history: Lista de mensagens anteriores [{role, content}, ...].
                 O content do assistant é o .tec gerado anteriormente.

    Returns:
        Conteúdo do ficheiro .tec gerado.
    """
    from smolagents.models import ChatMessage

    messages = [
        ChatMessage(role="system", content=[{"type": "text", "text": SYSTEM_PROMPT}])
    ]

    # Injeta histórico — limita a 6 turnos para não exceder o contexto
    for msg in history[-6:]:
        messages.append(
            ChatMessage(
                role=msg["role"],
                content=[{"type": "text", "text": msg["content"]}]
            )
        )

    # Adiciona o pedido atual
    messages.append(
        ChatMessage(role="user", content=[{"type": "text", "text": description}])
    )

    response = _MODEL(messages)

    if isinstance(response.content, list):
        return response.content[0].get("text", "")
    return response.content