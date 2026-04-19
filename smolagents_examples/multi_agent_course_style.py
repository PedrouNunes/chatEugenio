import os
import re
import sys

import requests
from markdownify import markdownify
from requests.exceptions import RequestException

# Adiciona a pasta raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smolagents import (
    CodeAgent,
    InferenceClientModel,
    WebSearchTool,
    tool,
)

from tools.calculate_cargo_travel_time import calculate_cargo_travel_time


@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()

        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Evita respostas gigantescas demais
        return markdown_content[:20000]

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def main():
    # Modelo/provider
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        provider="nscale",
        max_tokens=8096,
    )

    # Agente especializado em pesquisa
    web_agent = CodeAgent(
        tools=[
            WebSearchTool(),
            visit_webpage,
            calculate_cargo_travel_time,
        ],
        model=model,
        max_steps=10,
        name="web_agent",
        description=(
            "Browses the web to find Batman filming locations, supercar factories, "
            "coordinates, and supporting information."
        ),
        verbosity_level=1,
    )

    # Agente gerente/orquestrador
    manager_agent = CodeAgent(
        model=model,
        tools=[calculate_cargo_travel_time],
        managed_agents=[web_agent],
        additional_authorized_imports=[
            "time",
            "json",
            "math",
            "numpy",
            "pandas",
            "plotly",
            "plotly.express",
            "geopandas",
            "shapely",
        ],
        planning_interval=5,
        max_steps=15,
        verbosity_level=2,
    )

    task = """
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane
to here (we're in Gotham, 40.7128° N, 74.0060° W).

Also give me some supercar factories with similar cargo plane transfer time.
You need at least 6 points in total.

Return a structured report and, if possible, create a pandas dataframe with:
- name
- category
- latitude
- longitude
- travel_time_hours

Use the web_agent whenever web research is required.
"""

    result = manager_agent.run(task)

    print("\n=== FINAL RESPONSE ===\n")
    print(result)


if __name__ == "__main__":
    main()