import asyncio

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        message = "Hello, world!"
        return StopEvent(result=message)


async def main() -> None:
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run()
    print("=== RESULTADO DO WORKFLOW ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())