from langchain_community.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool


class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = (
        "Retrieves relevant party planning ideas from a local knowledge base. "
        "Use this for queries related to party decoration, entertainment, catering, "
        "luxury themes, or superhero events."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query related to party planning.",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        party_ideas = [
            {
                "text": (
                    "A superhero-themed masquerade ball with luxury decor, "
                    "including gold accents and velvet curtains."
                ),
                "source": "Party Ideas 1",
            },
            {
                "text": (
                    "Hire a professional DJ who can play themed music for "
                    "superheroes like Batman and Wonder Woman."
                ),
                "source": "Entertainment Ideas",
            },
            {
                "text": (
                    "For catering, serve dishes named after superheroes, like "
                    "The Hulk's Green Smoothie and Iron Man's Power Steak."
                ),
                "source": "Catering Ideas",
            },
            {
                "text": (
                    "Decorate with iconic superhero logos and projections of Gotham "
                    "and other superhero cities around the venue."
                ),
                "source": "Decoration Ideas",
            },
            {
                "text": (
                    "Interactive experiences with VR where guests can engage in "
                    "superhero simulations or compete in themed games."
                ),
                "source": "Entertainment Ideas",
            },
        ]

        source_docs = [
            Document(page_content=item["text"], metadata={"source": item["source"]})
            for item in party_ideas
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        docs_processed = text_splitter.split_documents(source_docs)

        self.retriever = BM25Retriever.from_documents(
            docs_processed,
            k=5,
        )

    def forward(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("The query must be a string.")

        docs = self.retriever.invoke(query)

        if not docs:
            return "No relevant ideas were found in the local knowledge base."

        results = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            results.append(
                f"===== Idea {i} =====\n"
                f"Source: {source}\n"
                f"{doc.page_content}\n"
            )

        return "\nRetrieved ideas:\n\n" + "\n".join(results)