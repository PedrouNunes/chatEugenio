import json

# TOOL
def get_weather(location):
    fake_data = {
        "New York": "18°C, cloudy",
        "Paris": "20°C, sunny",
        "Lisbon": "22°C, clear sky"
    }
    return fake_data.get(location, "Weather not found")


# AGENT COM CICLO COMPLETO
def agent(user_input):
    memory = []

    print("User:", user_input)

    # THOUGHT
    thought = "I need to get the weather information for the user."
    print("Thought:", thought)
    memory.append(thought)

    # ACTION
    action = {
        "action": "get_weather",
        "action_input": {"location": "Lisbon"}
    }

    print("Action:", json.dumps(action, indent=2))
    memory.append(action)

    # EXECUTE ACTION
    if action["action"] == "get_weather":
        location = action["action_input"]["location"]
        result = get_weather(location)

    # OBSERVATION
    observation = f"Weather result: {result}"
    print("Observation:", observation)
    memory.append(observation)

    # NOVO THOUGHT (com base na observation)
    new_thought = "Now I have the weather data, I can answer the user."
    print("Thought:", new_thought)
    memory.append(new_thought)

    # FINAL ANSWER
    final_answer = f"The weather in {location} is {result}"
    print("Final Answer:", final_answer)

    return memory


# RUN
agent("What's the weather today?")