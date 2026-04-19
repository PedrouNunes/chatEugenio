
from smolagents import CodeAgent, InferenceClientModel
import os
import sys

# Adiciona a pasta raiz do projeto ao path do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smolagents import CodeAgent, InferenceClientModel
from tools.party_planning_retriever import PartyPlanningRetrieverTool


def main():
    model = InferenceClientModel()

    local_retriever_tool = PartyPlanningRetrieverTool()

    agent = CodeAgent(
        model=model,
        tools=[local_retriever_tool],
        add_base_tools=False,
    )

    query = (
        "Find ideas for a luxury superhero-themed party, "
        "including entertainment, catering, and decoration options."
    )

    response = agent.run(query)

    print("\n=== FINAL RESPONSE ===\n")
    print(response)


if __name__ == "__main__":
    main()