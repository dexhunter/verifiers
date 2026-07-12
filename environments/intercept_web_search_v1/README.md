# intercept-web-search-v1

Small V1 example showing `@vf.intercept` over a provider-executed (server-side) tool.

## Develop

The built-in codex harness runs with native Responses `web_search` enabled
(`--harness.search true`). The task intercepts each assistant message before the harness sees
it, asks a rewrite model for a German version (the prompt reads the native web-search items off
`message.provider_state`), and returns the German text — Verifiers rewrites the native
Responses body and emits it back into codex's stream.

```text
codex --search
  -> V1 interception server
  -> OpenAI /v1/responses
  <- assistant message (web-search items on provider_state)
  <- @vf.intercept returns the German rewrite
  <- codex receives the rewritten Responses SSE
```

This example needs an OpenAI Responses endpoint that supports native `web_search`; the codex
harness installs its own pinned binary (a linux musl build, so use a container runtime off
linux).

```bash
uv run --with-editable . --with-editable environments/intercept_web_search_v1 \
  eval intercept-web-search-v1 \
  --harness.id codex --harness.search true --harness.runtime.type docker \
  -m gpt-4.1-mini \
  --client.base-url https://api.openai.com/v1 \
  --client.api-key-var OPENAI_API_KEY \
  -n 1 -r 1 --max-turns 2 --timeout.rollout 180
```

## Notes

The interceptor receives typed vf messages, not SSE bytes or provider JSON. For Responses
streams, Verifiers buffers the turn only when an interceptor is present, then emits a rewritten
stream after the replacement is committed to the trace. The replacement drops the native
web-search items — inspect them on `message.provider_state` before ruling.
