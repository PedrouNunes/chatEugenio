from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def catering_service_tool(query: str) -> str:
    """
    Return the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    best_service = max(services, key=services.get)
    return best_service

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_tokens=1024,
    temperature=0.5,
)

agent = CodeAgent(
    tools=[catering_service_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
)

result = agent.run(
    "Can you give me the name of the highest-rated catering service in Gotham City?"
)

print("\nRESULTADO FINAL:\n")
print(result)