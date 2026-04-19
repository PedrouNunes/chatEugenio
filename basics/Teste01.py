from typing import Callable

class Tool:
    def __init__(self, name: str, description: str, func: Callable, arguments: list, outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        return (
            f"Tool Name: {self.name}, "
            f"Description: {self.description}, "
            f"Arguments: {args_str}, "
            f"Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# Função que vira tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# Criando a tool manualmente
calculator_tool = Tool(
    "calculator",
    "Multiply two integers.",
    calculator,
    [("a", "int"), ("b", "int")],
    "int"
)

# Testando
print(calculator_tool.to_string())
print("Resultado:", calculator_tool(3, 5))