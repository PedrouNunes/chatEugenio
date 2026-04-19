# chat_api/agent.py
import os
import datetime
import pytz
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool, tool

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Get the current local time in a specified timezone.
    Args:
        timezone: A valid timezone string, e.g. 'Europe/Lisbon'.
    """
    try:
        tz = pytz.timezone(timezone)
        return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error: {str(e)}"

def build_agent() -> CodeAgent:
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=os.getenv("HF_TOKEN"),
        max_tokens=1024,
        temperature=0.5,
    )
    return CodeAgent(
        model=model,
        tools=[DuckDuckGoSearchTool(), get_current_time_in_timezone],
        max_steps=4,
        verbosity_level=0,
    )