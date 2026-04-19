from smolagents import tool

@tool
def FinalAnswerTool(answer: str) -> str:
    """Return the final answer to the user.
    
    Args:
        answer: The final response text.
    """
    return answer