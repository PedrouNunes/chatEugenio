from pathlib import Path

from basics.langGraph.langgraph_email_workflow_hf import build_graph


def main() -> None:
    graph = build_graph()
    png_data = graph.get_graph().draw_mermaid_png()

    output_path = Path("basics/langGraph/langgraph_email_workflow.png")
    output_path.write_bytes(png_data)

    print(f"Imagem gerada em: {output_path}")


if __name__ == "__main__":
    main()