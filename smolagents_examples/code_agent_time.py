from smolagents import CodeAgent, InferenceClientModel, tool
import datetime
import pytz

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    Get the current local time in a specified timezone.
    
    Args:
        timezone: A valid timezone string, for example 'Europe/Lisbon' or 'America/New_York'.
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_tokens=1024,
    temperature=0.5,
)

agent = CodeAgent(
    tools=[get_current_time_in_timezone],
    model=model,
    max_steps=4,
    verbosity_level=2,
)

result = agent.run("What time is it now in Europe/Lisbon?")
print("\nRESULTADO FINAL:\n")
print(result)