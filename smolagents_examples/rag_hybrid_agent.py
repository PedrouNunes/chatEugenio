import os
import sys

# Adiciona a pasta raiz do projeto ao path do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
from tools.party_planning_retriever import PartyPlanningRetrieverTool


def main():
    model = InferenceClientModel()

    search_tool = DuckDuckGoSearchTool()
    local_retriever_tool = PartyPlanningRetrieverTool()

    agent = CodeAgent(
        model=model,
        tools=[search_tool, local_retriever_tool],
        add_base_tools=False,
    )

    query = (
        "Create a luxury superhero-themed party plan using both current web trends "
        "and local party planning knowledge. Include decoration, entertainment, "
        "and catering suggestions."
    )

    response = agent.run(query)

    print("\n=== FINAL RESPONSE ===\n")
    print(response)


if __name__ == "__main__":
    main()