from llama_index.utils.workflow import draw_all_possible_flows

from basics.llama.llamaindex_workflow_loop import MultiStepWorkflow


def main() -> None:
    draw_all_possible_flows(MultiStepWorkflow, filename="basics/flow.html")
    print("Arquivo gerado: basics/flow.html")


if __name__ == "__main__":
    main()
    