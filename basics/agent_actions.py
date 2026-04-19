import json

# TOOL (função que o agent pode usar)
def get_weather(location):
    fake_data = {
        "New York": "18°C, cloudy",
        "Paris": "20°C, sunny",
        "Lisbon": "22°C, clear sky"
    }
    return fake_data.get(location, "Weather not found")


# AGENT (simulação simples)
def agent(user_input):
    print("User:", user_input)

    # THOUGHT
    print("Thought: I need to get the weather information.")

    # ACTION (formato JSON como no seu material)
    action = {
        "action": "get_weather",
        "action_input": {"location": "New York"}
    }

    print("Action:", json.dumps(action, indent=2))

    # PARSE + EXECUTE
    if action["action"] == "get_weather":
        location = action["action_input"]["location"]
        result = get_weather(location)

    # OBSERVATION
    print("Observation:", result)

    # FINAL ANSWER
    print("Final Answer:", f"The weather in {location} is {result}")


# RUN
agent("What's the weather in New York?")