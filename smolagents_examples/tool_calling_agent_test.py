from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, InferenceClientModel

model = InferenceClientModel(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="auto",
    max_tokens=1024,
    temperature=0.5,
)

agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=4,
)

result = agent.run("Search for the best music recommendations for a party at Wayne's mansion.")
print(result)