from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel


def main():
    # Tool de busca na web
    search_tool = DuckDuckGoSearchTool()

    # Modelo da Hugging Face
    model = InferenceClientModel()

    # Agent com capacidade de escrever/usar código e chamar tools
    agent = CodeAgent(
        model=model,
        tools=[search_tool],
        add_base_tools=False,
    )

    query = (
        "Search for luxury superhero-themed party ideas, "
        "including decorations, entertainment, and catering."
    )

    response = agent.run(query)

    print("\n=== FINAL RESPONSE ===\n")
    print(response)


if __name__ == "__main__":
    main()