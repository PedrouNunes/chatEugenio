import base64
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]


def build_llm() -> ChatOpenAI:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY não encontrado no .env. "
            "Adicione essa variável para rodar este exemplo."
        )

    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=openai_api_key,
    )


vision_llm = build_llm()


def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.
    """
    try:
        if not img_path:
            return "Nenhum caminho de imagem foi fornecido."

        if not os.path.exists(img_path):
            return f"Arquivo não encontrado: {img_path}"

        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        response = vision_llm.invoke(message)
        return str(response.content).strip()

    except Exception as e:
        error_msg = f"Erro ao extrair texto: {str(e)}"
        print(error_msg)
        return ""


def divide(a: int, b: int) -> float:
    """
    Divide a and b.
    """
    if b == 0:
        raise ValueError("Não é possível dividir por zero.")
    return a / b


tools = [divide, extract_text]

llm = build_llm()
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


def assistant(state: AgentState):
    textual_description_of_tool = """
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (string).

    Returns:
        A single string containing the extracted text from the image.

divide(a: int, b: int) -> float:
    Divide a and b.
"""

    image = state["input_file"]

    sys_msg = SystemMessage(
        content=(
            "You are a helpful butler named Alfred who serves Mr. Wayne. "
            "You can analyze documents and run computations with the provided tools.\n\n"
            f"Available tools:\n{textual_description_of_tool}\n"
            f"Currently loaded image: {image}\n\n"
            "Rules:\n"
            "- If the user asks about the content of the provided image, use extract_text.\n"
            "- If the user asks for a calculation, use divide when appropriate.\n"
            "- After receiving tool results, answer clearly and concisely.\n"
        )
    )

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"],
    }


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()


def print_messages(result: dict) -> None:
    print("\n=== HISTÓRICO DA EXECUÇÃO ===\n")
    for m in result["messages"]:
        m.pretty_print()
        print()


def run_calculation_example(react_graph) -> None:
    print("\n===== EXEMPLO 1: CÁLCULO =====\n")

    messages = [HumanMessage(content="Divide 6790 by 5")]
    result = react_graph.invoke(
        {
            "messages": messages,
            "input_file": None,
        }
    )

    print_messages(result)


def run_document_example(react_graph) -> None:
    print("\n===== EXEMPLO 2: ANÁLISE DE DOCUMENTO =====\n")

    image_path = "basics/data/Batman_training_and_meals.png"

    messages = [
        HumanMessage(
            content=(
                "According to the note provided by Mr. Wayne in the provided image, "
                "what items should I buy for tomorrow's dinner menu?"
            )
        )
    ]

    result = react_graph.invoke(
        {
            "messages": messages,
            "input_file": image_path,
        }
    )

    print_messages(result)


def main() -> None:
    react_graph = build_graph()

    run_calculation_example(react_graph)
    run_document_example(react_graph)


if __name__ == "__main__":
    main()