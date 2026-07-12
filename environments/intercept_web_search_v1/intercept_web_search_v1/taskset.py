"""intercept-web-search: rewrite what a provider-executed tool feeds the conversation.

Codex answers with its native OpenAI `web_search` results; the task's `@vf.intercept`
sees each assistant turn before the harness does and replaces it with a German rewrite
from a judge model. The native web-search items are inspectable on
`message.provider_state` (the rewrite prompt reads them); the replacement drops them —
the harness receives only the German answer, and the trace records it as the model's
own turn.
"""

from pydantic import Field

import verifiers.v1 as vf

PROMPT = (
    "Use web search to inspect the front page of example.com. "
    "Answer in one sentence and include a markdown source link."
)
GERMAN_EXAMPLE_URL = "https://de.wikipedia.org/wiki/Example.com"


class GermanRewrite(vf.StrictBaseModel):
    text_de: str


class WebSearchData(vf.TaskData):
    expected_url: str


class InterceptWebSearchTaskConfig(vf.TaskConfig):
    rewrite: vf.JudgeConfig = Field(
        default_factory=lambda: vf.JudgeConfig(
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key_var="OPENAI_API_KEY",
        )
    )
    """Model config for the interceptor's German rewrite call."""


class InterceptWebSearchTask(
    vf.Task[WebSearchData, vf.State, InterceptWebSearchTaskConfig]
):
    @vf.intercept
    async def german_rewrite(
        self, message: vf.AssistantMessage, trace: vf.Trace
    ) -> str | None:
        if message.tool_calls or not message.content:
            return None
        judge = vf.Judge(self.config.rewrite)
        result = await judge.complete(
            (
                "Rewrite the assistant answer into natural German. Return JSON with "
                "text_de. Keep the factual meaning and include exactly one markdown "
                f"source link, preferring {GERMAN_EXAMPLE_URL} for Example.com.\n\n"
                f"Original answer:\n{message.content}\n\n"
                f"Web-search items:\n{message.provider_state or []}"
            ),
            trace=trace,
            schema=GermanRewrite,
            temperature=0,
        )
        rewrite = result.parsed
        if rewrite is None:
            raise RuntimeError("rewrite model returned no GermanRewrite object")
        trace.info["intercept_rewrite"] = rewrite.model_dump()
        return rewrite.text_de

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def saw_german_rewrite(self, trace: vf.Trace) -> float:
        reply = trace.last_reply or ""
        return float(self.data.expected_url in reply and "Die " in reply)


class InterceptWebSearchConfig(vf.TasksetConfig):
    task: InterceptWebSearchTaskConfig = InterceptWebSearchTaskConfig()


class InterceptWebSearchTaskset(
    vf.Taskset[InterceptWebSearchTask, InterceptWebSearchConfig]
):
    def load(self) -> list[InterceptWebSearchTask]:
        return [
            InterceptWebSearchTask(
                WebSearchData(idx=0, prompt=PROMPT, expected_url=GERMAN_EXAMPLE_URL),
                self.config.task,
            )
        ]
