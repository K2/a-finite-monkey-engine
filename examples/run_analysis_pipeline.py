# run_analysis_pipeline.py
import asyncio
from finite_monkey.agents.researcher import Researcher


async def main():
    researcher = Researcher()
    await researcher.analyze_code_async()
    # Your analysis pipeline logic here
if __name__ == "__main__":
    asyncio.run(main())
    main()
