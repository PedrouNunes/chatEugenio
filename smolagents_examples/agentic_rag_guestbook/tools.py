import random
from huggingface_hub import list_models
from smolagents import Tool

# Compatibilidade entre versões do smolagents
try:
    from smolagents import WebSearchTool

    search_tool = WebSearchTool()
    SEARCH_TOOL_NAME = "WebSearchTool"
except ImportError:
    from smolagents import DuckDuckGoSearchTool

    search_tool = DuckDuckGoSearchTool()
    SEARCH_TOOL_NAME = "DuckDuckGoSearchTool"


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str) -> str:
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20},
        ]
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author or organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str) -> str:
        try:
            models = list(
                list_models(
                    author=author,
                    sort="downloads",
                    direction=-1,
                    limit=1,
                )
            )

            if models:
                model = models[0]
                downloads = model.downloads if model.downloads is not None else 0
                return (
                    f"The most downloaded model by {author} is "
                    f"{model.id} with {downloads:,} downloads."
                )

            return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"


class LatestNewsTool(Tool):
    name = "latest_news"
    description = "Searches the web for the latest news about a specific topic."
    inputs = {
        "topic": {
            "type": "string",
            "description": "The topic to search recent news about."
        }
    }
    output_type = "string"

    def forward(self, topic: str) -> str:
        try:
            query = f"latest news about {topic}"
            result = search_tool(query)
            return f"Latest news results for '{topic}':\n{result}"
        except Exception as e:
            return f"Error fetching latest news for {topic}: {str(e)}"


weather_info_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()
latest_news_tool = LatestNewsTool()


if __name__ == "__main__":
    print(f"=== TESTE DAS TOOLS ({SEARCH_TOOL_NAME}) ===\n")

    print("1) Web Search Tool")
    try:
        search_result = search_tool("Who's the current President of France?")
        print(search_result)
    except Exception as e:
        print(f"Erro no search tool: {e}")

    print("\n2) Weather Tool")
    print(weather_info_tool("Paris"))

    print("\n3) Hub Stats Tool")
    print(hub_stats_tool("facebook"))

    print("\n4) Latest News Tool")
    print(latest_news_tool("artificial intelligence"))