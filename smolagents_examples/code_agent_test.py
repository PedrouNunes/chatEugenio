from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggest a menu based on the occasion.
    
    Args:
        occasion: The type of occasion for the party. Use values like 'casual', 'formal', or 'superhero'.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_tokens=1024,
    temperature=0.5,
)

agent = CodeAgent(
    tools=[suggest_menu],
    model=model,
    max_steps=4,
    verbosity_level=2,
)

result = agent.run("Prepare a formal menu for the party.")
print("\nRESULTADO FINAL:\n")
print(result)