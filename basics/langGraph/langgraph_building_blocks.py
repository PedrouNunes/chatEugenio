import random
from typing import Literal

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    graph_state: str


def node_1(state: State) -> dict:
    print("---Node 1---")
    return {"graph_state": state["graph_state"] + " I am"}


def node_2(state: State) -> dict:
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + " happy!"}


def node_3(state: State) -> dict:
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " sad!"}


def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    user_input = state["graph_state"]

    print(f"Estado atual antes da decisão: {user_input}")

    if random.random() < 0.5:
        return "node_2"

    return "node_3"


def build_graph():
    builder = StateGraph(State)

    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

    graph = builder.compile()
    return graph


def main() -> None:
    graph = build_graph()

    result = graph.invoke({"graph_state": "Hi, this is Pedro."})

    print("\n=== RESULTADO FINAL ===")
    print(result)


if __name__ == "__main__":
    main()