import datasets
from llama_index.core.schema import TextNode
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever


def load_guest_nodes():
    """
    Carrega o dataset do curso e converte cada convidado em um TextNode.
    Usamos TextNode porque o BM25Retriever atual trabalha naturalmente com nodes.
    """
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

    nodes = [
        TextNode(
            text="\n".join(
                [
                    f"Name: {guest['name']}",
                    f"Relation: {guest['relation']}",
                    f"Description: {guest['description']}",
                    f"Email: {guest['email']}",
                ]
            ),
            metadata={"name": guest["name"]},
        )
        for guest in guest_dataset
    ]

    return nodes


# Carrega os nodes uma única vez
guest_nodes = load_guest_nodes()

# Cria o retriever BM25
bm25_retriever = BM25Retriever.from_defaults(
    nodes=guest_nodes,
    similarity_top_k=3,
)


def get_guest_info_retriever(query: str) -> str:
    """
    Retrieves detailed information about gala guests based on their name or relation.
    """
    results = bm25_retriever.retrieve(query)

    if not results:
        return "No matching guest information found."

    formatted_results = []
    for result in results[:3]:
        node = result.node
        formatted_results.append(node.text)

    return "\n\n".join(formatted_results)


# Inicializa a tool para ser usada depois no app.py
guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever)


if __name__ == "__main__":
    test_query = "Lady Ada Lovelace"
    print("=== TESTE DO RETRIEVER ===")
    print(f"Consulta: {test_query}\n")
    print(get_guest_info_retriever(test_query))