# chat_api/eugenio.py
import os
import random
import datasets
from huggingface_hub import list_models
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool, Tool


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {"location": {"type": "string", "description": "The location to get weather information for."}}
    output_type = "string"

    def forward(self, location: str) -> str:
        conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20},
        ]
        data = random.choice(conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {"author": {"type": "string", "description": "The username of the model author or organization."}}
    output_type = "string"

    def forward(self, author: str) -> str:
        try:
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
            if models:
                model = models[0]
                downloads = model.downloads if model.downloads is not None else 0
                return f"The most downloaded model by {author} is {model.id} with {downloads:,} downloads."
            return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"


class LatestNewsTool(Tool):
    name = "latest_news"
    description = "Searches the web for the latest news about a specific topic."
    inputs = {"topic": {"type": "string", "description": "The topic to search recent news about."}}
    output_type = "string"

    def __init__(self, search_tool):
        super().__init__()
        self.search_tool = search_tool

    def forward(self, topic: str) -> str:
        try:
            result = self.search_tool(f"latest news about {topic}")
            return f"Latest news about '{topic}':\n{result}"
        except Exception as e:
            return f"Error fetching news for {topic}: {str(e)}"


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {"query": {"type": "string", "description": "The name or relation of the guest."}}
    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str) -> str:
        results = self.retriever.invoke(query)
        if not results:
            return "No matching guest information found."
        return "\n\n".join(doc.page_content for doc in results[:3])


# Builder 
def load_guest_documents():
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
    return [
        Document(
            page_content="\n".join([
                f"Name: {g['name']}",
                f"Relation: {g['relation']}",
                f"Description: {g['description']}",
                f"Email: {g['email']}",
            ]),
            metadata={"name": g["name"]},
        )
        for g in guest_dataset
    ]


def build_alfred() -> CodeAgent:
    hf_token = os.getenv("HF_TOKEN")

    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=hf_token,
        max_tokens=1024,
        temperature=0.5,
    )

    search_tool = WebSearchTool()
    docs = load_guest_documents()

    alfred = CodeAgent(
        tools=[
            GuestInfoRetrieverTool(docs),
            search_tool,
            WeatherInfoTool(),
            HubStatsTool(),
            LatestNewsTool(search_tool),
        ],
        model=model,
        add_base_tools=True,
        planning_interval=3,
        verbosity_level=0,
    )

    return alfred