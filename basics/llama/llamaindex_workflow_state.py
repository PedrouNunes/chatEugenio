import asyncio

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step


class StatefulWorkflow(Workflow):
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
        await ctx.store.set("query", "What is the capital of France?")

        query = await ctx.store.get("query")
        result = f"A consulta armazenada no contexto foi: {query}"

        return StopEvent(result=result)


async def main() -> None:
    workflow = StatefulWorkflow(timeout=10, verbose=False)
    ctx = Context(workflow)

    result = await workflow.run(ctx=ctx)

    print("=== RESULTADO DO WORKFLOW COM ESTADO ===")
    print(result)

    saved_query = await ctx.store.get("query")
    print("\n=== VALOR NO CONTEXTO ===")
    print(saved_query)


if __name__ == "__main__":
    asyncio.run(main())