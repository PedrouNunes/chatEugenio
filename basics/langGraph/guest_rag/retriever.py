import datasets
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool


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


# Carrega os documentos uma vez
docs = load_guest_documents()

# Cria o retriever BM25
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3


def extract_text(query: str) -> str:
    """
    Retrieves detailed information about gala guests based on their name or relation.
    """
    results = bm25_retriever.invoke(query)

    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])

    return "No matching guest information found."


guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation.",
)


if __name__ == "__main__":
    test_query = "Lady Ada Lovelace"

    print("=== TESTE DO RETRIEVER ===")
    print(f"Consulta: {test_query}\n")
    print(extract_text(test_query))