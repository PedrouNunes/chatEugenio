import random
from huggingface_hub import list_models
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool


search_tool = DuckDuckGoSearchRun()


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
        return search_tool.invoke(f"latest news about {topic}")
    except Exception as e:
        return f"Error fetching latest news for {topic}: {str(e)}"


weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location.",
)

hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub.",
)

latest_news_tool = Tool(
    name="get_latest_news",
    func=get_latest_news,
    description="Searches the web for the latest news about a specific topic.",
)


if __name__ == "__main__":
    print("=== TESTE DAS TOOLS (LangChain/LangGraph) ===\n")

    print("1) Web Search Tool")
    try:
        print(search_tool.invoke("Who's the current President of France?"))
    except Exception as e:
        print(f"Erro no search tool: {e}")

    print("\n2) Weather Tool")
    print(weather_info_tool.invoke("Paris"))

    print("\n3) Hub Stats Tool")
    print(hub_stats_tool.invoke("facebook"))

    print("\n4) Latest News Tool")
    print(latest_news_tool.invoke("artificial intelligence"))