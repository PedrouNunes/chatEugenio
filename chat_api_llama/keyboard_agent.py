from smolagents import LiteLLMModel, tool

OLLAMA_MODEL = "ollama/qwen2.5-coder:3b"

_MODEL = LiteLLMModel(
    model_id=OLLAMA_MODEL,
    api_base="http://localhost:11434",
    max_tokens=1000,
    temperature=0.1,
)

# Prompt de sistema com as regras do formato .tec e exemplos reais.
# O modelo aprende o formato apenas a partir destes exemplos (few-shot).
SYSTEM_PROMPT = """
You are an assistant that creates keyboard files (.tec) for the Eugénio AAC system.
Generate a valid .tec file based on the description. Return ONLY the file content.

--- FORMAT RULES ---

Each section: LINHA name  then  GRUPO name  (same name)
Spaces in names use ;;; separator  (ex: "apagar letra" -> "apagar;;;letra")
First key of every section must be: TECLA TECLA_VAZIA
Regular keys: TECLA TECLA_NORMAL [image] [label] [value] 1 -1 -1
Prediction keys: TECLA TECLA_PREDICAO  or  TECLA TECLA_PREDICAO_FRASE

--- ACTION BUTTONS (copy exactly) ---

Keyboard shortcuts:
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

Eugénio system actions:
  TECLA TECLA_NORMAL ... Mais;;;predições;;;de;;;vocabulário [Show-Next-Predictions] 1 -1 -1
  TECLA TECLA_NORMAL <---> Limpar;;;editor;;;de;;;mensagens [Clear-Editing-Area] 1 -1 -1
  TECLA TECLA_NORMAL <---> Escrever;;;em;;;aplicação;;;externa [Send-Message-To-External-Editor] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;palavra [Synthesize-Word-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;frase [Synthesize-Sentence-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Sintetizar;;;todo;;;o;;;texto [Synthesize-All-Text-To-Speech] 1 -1 -1
  TECLA TECLA_NORMAL <---> Desligar [Quit-Application] 1 -1 -1

--- EXAMPLE (real keyboard file) ---

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
TECLA TECLA_NORMAL Ç ç ç 1 -1 -1
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

--- IMPORTANT ---
Write accented chars directly: ç ã ã é ó — never use \\xe7 or similar escapes.
Return ONLY the .tec content, nothing else.
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
    from smolagents.models import ChatMessage

    messages = [
        ChatMessage(role="system", content=[{"type": "text", "text": SYSTEM_PROMPT}])
    ]

    for msg in history[-6:]:
        messages.append(
            ChatMessage(
                role=msg["role"],
                content=[{"type": "text", "text": msg["content"]}]
            )
        )

    messages.append(
        ChatMessage(role="user", content=[{"type": "text", "text": description}])
    )

    response = _MODEL(messages)

    if isinstance(response.content, list):
        return response.content[0].get("text", "")
    return response.content