system({
    title: "A planner agent  python",
})

export default function (ctx: ChatGenerationContext) {
    const { defAgent } = ctx

    defAgent(
        "planner,   You are an expert coder in Python.",
        "generates a plan to solve a task",
        `Generate a detailed plan as a list of tasks so that a smaller LLM can use agents to execute the plan.`,
        {
            model: "reasoning",
            system: [
                "system.assistant",
                "system.planner",
                "system.safety_jailbreak",
                "system.safety_harmful_content",
            ],
        }
    )
}
