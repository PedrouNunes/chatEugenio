import datasets
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool


def load_guest_documents():
    """
    Carrega o dataset do curso e converte cada convidado em um Document.
    """
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

    docs = [
        Document(
            page_content="\n".join(
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

    return docs


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = (
        "Retrieves detailed information about gala guests based on their name or relation."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        self.docs = docs
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str) -> str:
        results = self.retriever.invoke(query)

        if not results:
            return "No matching guest information found."

        top_results = results[:3]
        return "\n\n".join(doc.page_content for doc in top_results)


def build_guest_info_tool():
    """
    Função auxiliar para construir a tool pronta para uso no app.py.
    """
    docs = load_guest_documents()
    return GuestInfoRetrieverTool(docs)


if __name__ == "__main__":
    docs = load_guest_documents()
    tool = GuestInfoRetrieverTool(docs)

    test_query = "Ada Lovelace"
    result = tool.forward(test_query)

    print("=== TESTE DO RETRIEVER ===")
    print(f"Consulta: {test_query}\n")
    print(result)