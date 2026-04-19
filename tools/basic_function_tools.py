from llama_index.core.tools import FunctionTool


def get_weather(location: str) -> str:
    """
    Useful for getting the weather for a given location.
    """
    print(f"Consultando clima para: {location}")
    return f"O clima em {location} está ensolarado."


def calculate_dinner_budget_per_person(total_budget: float, guest_count: int) -> str:
    """
    Useful for calculating the budget per person for an event.
    """
    if guest_count <= 0:
        return "A quantidade de convidados deve ser maior que zero."

    value = total_budget / guest_count
    return (
        f"Com um orçamento total de {total_budget:.2f} euros "
        f"para {guest_count} convidados, o valor por pessoa é {value:.2f} euros."
    )


def main() -> None:
    weather_tool = FunctionTool.from_defaults(
        fn=get_weather,
        name="weather_lookup_tool",
        description="Útil para obter o clima de uma localização específica.",
    )

    budget_tool = FunctionTool.from_defaults(
        fn=calculate_dinner_budget_per_person,
        name="dinner_budget_tool",
        description=(
            "Útil para calcular o orçamento por pessoa em um jantar ou evento."
        ),
    )

    print("\n=== TESTE 1: WEATHER TOOL ===")
    weather_result = weather_tool.call(location="New York")
    print(weather_result)

    print("\n=== TESTE 2: BUDGET TOOL ===")
    budget_result = budget_tool.call(total_budget=60.0, guest_count=4)
    print(budget_result)

    print("\n=== METADADOS DAS TOOLS ===")
    print(f"Nome: {weather_tool.metadata.name}")
    print(f"Descrição: {weather_tool.metadata.description}")
    print()
    print(f"Nome: {budget_tool.metadata.name}")
    print(f"Descrição: {budget_tool.metadata.description}")


if __name__ == "__main__":
    main()