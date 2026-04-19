import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    model="moonshotai/Kimi-K2.5",
    token=HF_TOKEN
)

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}

example use:

{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB

Observation: the result of the action.

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

def get_weather(location):
    return f"The weather in {location} is sunny with low temperatures."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]

first_output = client.chat.completions.create(
    messages=messages,
    max_tokens=150,
    stop=["Observation:"],
    extra_body={"thinking": {"type": "disabled"}},
)

assistant_text = first_output.choices[0].message.content
print("FIRST MODEL OUTPUT:\n")
print(assistant_text)
print("\n" + "=" * 60 + "\n")

messages.append({
    "role": "assistant",
    "content": assistant_text + "\nObservation:\n" + get_weather("London")
})

second_output = client.chat.completions.create(
    messages=messages,
    max_tokens=200,
    extra_body={"thinking": {"type": "disabled"}},
)

print("FINAL MODEL OUTPUT:\n")
print(second_output.choices[0].message.content)