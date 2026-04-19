import asyncio
import random

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step


class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(
        self, ev: StartEvent | LoopEvent
    ) -> ProcessingEvent | LoopEvent:
        if isinstance(ev, LoopEvent):
            print(f"Recebi LoopEvent: {ev.loop_output}")

        if random.randint(0, 1) == 0:
            print("Algo deu errado. Voltando para o início...")
            return LoopEvent(loop_output="Tentando novamente a partir do step_one.")
        else:
            print("Tudo deu certo no step_one.")
            return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


async def main() -> None:
    workflow = MultiStepWorkflow(timeout=20, verbose=False)
    result = await workflow.run()
    print("\n=== RESULTADO FINAL ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())