import random
from huggingface_hub import list_models
from llama_index.core.tools import FunctionTool
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec


duckduckgo_spec = DuckDuckGoSearchToolSpec()
search_tool = FunctionTool.from_defaults(
    duckduckgo_spec.duckduckgo_full_search,
    name="web_search",
    description="Search the web for recent or factual information.",
)


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            downloads = model.downloads if model.downloads is not None else 0
            return f"The most downloaded model by {author} is {model.id} with {downloads:,} downloads."

        return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"


def get_latest_news(topic: str) -> str:
    """Searches the web for the latest news about a specific topic."""
    try:
        result = duckduckgo_spec.duckduckgo_full_search(f"latest news about {topic}")
        if isinstance(result, list):
            snippets = []
            for item in result[:5]:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    body = item.get("body", "")
                    href = item.get("href", "")
                    snippets.append(f"Title: {title}\nBody: {body}\nLink: {href}")
                else:
                    snippets.append(str(item))
            return "\n\n".join(snippets)

        return str(result)
    except Exception as e:
        return f"Error fetching latest news for {topic}: {str(e)}"


weather_info_tool = FunctionTool.from_defaults(get_weather_info)
hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)
latest_news_tool = FunctionTool.from_defaults(get_latest_news)


if __name__ == "__main__":
    print("=== TESTE DAS TOOLS (LlamaIndex) ===\n")

    print("1) Web Search Tool")
    try:
        response = search_tool("Who's the current President of France?")
        print(response)
    except Exception as e:
        print(f"Erro no search tool: {e}")

    print("\n2) Weather Tool")
    print(get_weather_info("Paris"))

    print("\n3) Hub Stats Tool")
    print(get_hub_stats("facebook"))

    print("\n4) Latest News Tool")
    print(get_latest_news("artificial intelligence"))