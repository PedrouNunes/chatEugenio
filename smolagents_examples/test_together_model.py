from smolagents import InferenceClientModel

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    provider="together",
    max_tokens=512,
)

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Say hello in one short sentence."}],
    }
]

response = model(messages)
print(response)