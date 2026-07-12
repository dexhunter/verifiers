"""The interception server: harness chat-completions, caught and proxied.

Every rollout runs an harness program whose OpenAI-style calls are caught here: a small
localhost server routes each `POST /v1/chat/completions` to our `Client`, records the turn
into the trace's message graph, and returns the result in OpenAI shape. We inject
`OPENAI_BASE_URL`/`OPENAI_API_KEY` so the program's SDK talks to us. Both non-streaming and
SSE requests are supported.

One server multiplexes many rollouts: each rollout registers a `RolloutSession` under its
own secret (the bearer token the harness already sends), and the server routes by that
secret to the right session. So N rollouts need one server (and, behind a remote runtime,
one tunnel) per pool member rather than one each — see `interception.pool`.

When a rollout sets a user simulator (see `verifiers.v1.mcp.user`), the session also drives it:
after each model turn it injects the simulator's reply as a user turn and re-prompts the
model, so a multi-turn exchange plays out within one program request, transparently to the
harness. When the row carries no prompt (`TaskData.prompt is None`), the simulator also
opens the conversation: its first turn is seeded before the model is ever called. Tools are
handled out-of-band (run by the harness).
"""

import asyncio
import contextlib
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING, get_args, get_type_hints

from aiohttp import web
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, from_json, to_json

from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import invoke
from verifiers.v1.dialects import DIALECTS, Dialect
from verifiers.v1.dialects.base import is_sse_done_event
from verifiers.v1 import graph
from verifiers.v1.errors import (
    OverlongPromptError,
    RolloutError,
    TaskError,
    UserError,
)
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Message,
    MessageContent,
    Messages,
    Tool,
    ToolMessage,
)

if TYPE_CHECKING:
    from verifiers.v1.mcp import Respond

logger = logging.getLogger(__name__)


# Each session proxies one rollout's own harness requests, so aiohttp's default 1 MiB body
# cap is an artificial bottleneck — a large tool result (e.g. a `cat` of a big file) trips it
# and the harness gets a 413. Allow large bodies; the upstream provider and the model's
# context window are the real limits, this is just a host-OOM backstop.
_MAX_REQUEST_BODY = 1024**3  # 1 GiB (aiohttp's default is 1 MiB)
_KEEPALIVE_INTERVAL_SECONDS = 3
_STREAM_QUEUE_MAXSIZE = 16
_STREAM_MEMORY_BUFFER = 8 * 1024**2
# The server binds loopback; callers reach it via localhost or a host tunnel (see `reachable_url`).
_HOST = "127.0.0.1"


def _completion_response(completion: dict | None) -> web.Response:
    """Serialize a model's JSON-native response without an intermediate string."""
    try:
        body = to_json(completion, inf_nan_mode="constants")
    except PydanticSerializationError:
        return web.json_response(completion)
    return web.Response(body=body, content_type="application/json", charset="utf-8")


async def _queue_chunks(
    chunks: AsyncIterator[bytes],
    queue: asyncio.Queue[bytes | None],
    ready: asyncio.Event,
) -> None:
    try:
        async for chunk in chunks:
            await queue.put(chunk)
            ready.set()
    finally:
        await queue.put(None)
        ready.set()


@dataclass(frozen=True)
class RolloutLimits:
    """Per-rollout framework limits (None = no cap), checked before each turn is served.
    The first limit reached refuses the turn — halting any harness, the same mechanism as
    a @stop — and becomes the trace's stop condition. Each caps a trace computed property:
    `max_turns` -> num_turns, `max_input_tokens` -> num_input_tokens, `max_output_tokens` ->
    num_output_tokens, `max_total_tokens` -> num_total_tokens. Token caps are soft by one turn:
    they're checked between turns, so the turn that crosses a cap still completes."""

    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None

    def reached(self, trace: Trace) -> str | None:
        """The name of the first limit `trace` has reached, or None if within all caps."""
        if self.max_turns is not None and trace.num_turns >= self.max_turns:
            return "max_turns"
        if (
            self.max_input_tokens is not None
            and trace.num_input_tokens >= self.max_input_tokens
        ):
            return "max_input_tokens"
        if (
            self.max_output_tokens is not None
            and trace.num_output_tokens >= self.max_output_tokens
        ):
            return "max_output_tokens"
        if (
            self.max_total_tokens is not None
            and trace.num_total_tokens >= self.max_total_tokens
        ):
            return "max_total_tokens"
        return None


@dataclass
class RolloutSession:
    ctx: ModelContext
    trace: Trace
    stops: list[Callable[[Trace], Awaitable[bool]]] = field(default_factory=list)
    limits: RolloutLimits = field(default_factory=RolloutLimits)
    intercepts: list[Callable[..., Awaitable[object]]] = field(default_factory=list)
    rewritten_response_ids: set[str] = field(default_factory=set)
    user: "Respond | None" = None
    """A user simulator the rollout sets before the harness runs (see `verifiers.v1.mcp.user`).
    When set, each model turn with no tool call is followed by the simulator's reply,
    injected as a user turn, and the model is re-prompted — all within one program request,
    transparently to the harness."""
    opening: Messages | None = None
    """Cached opening `respond("")` messages for a no-prompt task. Computed once and re-injected on
    every request until the first turn lands on the trace — so a retried opening request (e.g. the
    harness SDK retrying a transient model 502, before any turn is recorded) never calls `respond`
    twice and advances the simulator's queue past the opening."""
    error: "RolloutError | None" = None
    """The latest unresolved model-call failure. The harness only sees it as an HTTP error
    (and may swallow it, or exit non-zero), so the rollout re-raises this original error once the
    harness returns — recording the real `ProviderError` instead of a secondary `HarnessError`.
    Reset before each model turn, so a successful retry clears it."""

    async def intercept(self, message: Message) -> Message | None:
        """Return the first interceptor replacement, or None for native pass-through."""
        try:
            for interceptor in self.intercepts:
                # The `message` annotation selects which messages an interceptor sees:
                # `vf.AssistantMessage` / `vf.ToolMessage` filter, anything else sees all.
                hint = get_type_hints(interceptor).get("message")
                if hint is not None and not isinstance(message, get_args(hint) or hint):
                    continue
                replacement = await invoke(
                    interceptor,
                    {"message": message.model_copy(deep=True), "trace": self.trace},
                )
                if replacement is None:
                    continue
                if isinstance(replacement, str):
                    # A string replaces the whole assistant turn — dropping its tool calls —
                    # while a tool result keeps its identity fields and swaps only content.
                    replacement = (
                        AssistantMessage(content=replacement)
                        if isinstance(message, AssistantMessage)
                        else message.model_copy(update={"content": replacement})
                    )
                if type(replacement) is not type(message):
                    raise TaskError(
                        f"@intercept for {type(message).__name__} returned "
                        f"{type(replacement).__name__}"
                    )
                if isinstance(replacement, AssistantMessage):
                    # The rewrite carries no sampled identity: native provider state and
                    # plain-text reasoning are dropped so committed == replayed everywhere.
                    replacement = replacement.model_copy(
                        update={"provider_state": None, "reasoning_content": None}
                    )
                return replacement
        except RolloutError:
            raise
        except Exception as e:
            raise TaskError(f"@intercept failed: {type(e).__name__}: {e}") from e
        return None

    async def refused(self) -> str | None:
        """The framework's limits (turns / token budget) and `@stop` checks, run before each
        model call. Sets the stop condition and returns its name, else None. A refused first
        call halts the harness (its model call errors out); Harness.run treats it as clean. A task
        that ends a trajectory from `trace.state` does it with its own `@stop` (run here generically),
        so the interception server holds no opinion about the state's contents."""
        if (limit := self.limits.reached(self.trace)) is not None:
            self.trace.stop(limit)
            logger.debug("limit %r reached: id=%s", limit, self.trace.id)
            return limit
        for stop in self.stops:
            if await stop(self.trace):
                self.trace.stop(stop.__name__)
                logger.debug("stop %r fired: id=%s", stop.__name__, self.trace.id)
                return stop.__name__
        return None


class InterceptionServer:
    def __init__(self) -> None:
        self.sessions: dict[str, RolloutSession] = {}
        self.port = 0
        self.runner: web.AppRunner | None = None

    def register(self, session: RolloutSession) -> str:
        """Add a session under a fresh secret (the bearer token the harness must send) and
        return it."""
        secret = secrets.token_urlsafe(16)
        self.sessions[secret] = session
        return secret

    def unregister(self, secret: str) -> None:
        self.sessions.pop(secret, None)

    def _handler_for(self, dialect: Dialect):
        """Bind a route's dialect to the request handler — the route the SDK posts to is what
        selects the wire format (see `dialects.DIALECTS`)."""

        async def handler(request: web.Request) -> web.StreamResponse:
            return await self.handle_request(request, dialect)

        return handler

    def _aux_handler_for(self, dialect: Dialect, route: str):
        async def handler(request: web.Request) -> web.Response:
            return await self.handle_aux(request, dialect, route)

        return handler

    async def __aenter__(self) -> "InterceptionServer":
        app = web.Application(client_max_size=_MAX_REQUEST_BODY)
        for dialect in DIALECTS:
            for route in dialect.routes:
                app.router.add_post(route, self._handler_for(dialect))
            for aux in dialect.aux_routes:
                app.router.add_post(aux, self._aux_handler_for(dialect, aux))
        # The shared-state back-channel (see `verifiers.v1.state`): a rollout's tool/user servers
        # GET/PUT their `self.state` here, keyed by the same bearer secret as the model routes.
        app.router.add_get("/state", self.handle_state_get)
        app.router.add_put("/state", self.handle_state_put)
        # A launched tool/user server fetches its rollout's task here to run `setup_task` — the task
        # is never passed via env, only over this channel, keyed by the same bearer secret.
        app.router.add_get("/task", self.handle_task_get)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, _HOST, 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # actual ephemeral port
        logger.info("interception up: url=http://%s:%d", _HOST, self.port)
        return self

    async def __aexit__(self, *exc) -> None:
        logger.info("interception down: url=http://%s:%d", _HOST, self.port)
        if self.runner is not None:
            await self.runner.cleanup()

    def _fail(
        self, session: RolloutSession, dialect: Dialect, error: RolloutError
    ) -> web.Response:
        """Stash a model-turn-adjacent failure (a `@stop` or user simulator raising) so the rollout
        re-raises it as the real cause, and report it to the harness as an HTTP error."""
        session.error = error
        logger.warning(
            "rollout %s failed: %s: %s", session.trace.id, type(error).__name__, error
        )
        return web.json_response(
            dialect.error_body(str(error)),
            status=getattr(error, "status_code", 502),
        )

    async def handle_request(
        self, request: web.Request, dialect: Dialect
    ) -> web.StreamResponse:
        session = self.sessions.get(dialect.secret(request.headers))
        if session is None:
            logger.warning("interception: unauthorized request")
            return web.json_response(dialect.error_body("unauthorized"), status=401)
        raw = await request.read()
        try:
            body = from_json(raw)
        except ValueError:
            body = json.loads(raw)
        # Keep `read()` for aiohttp's size guard, then release its cache and our local
        # alias after parsing so the wire body does not survive model inference.
        request._read_bytes = None
        del raw
        if session.intercepts:
            if body.get("previous_response_id") in session.rewritten_response_ids:
                return self._fail(
                    session,
                    dialect,
                    TaskError(
                        "@intercept cannot safely continue a rewritten response by "
                        "previous_response_id; replay the rewritten output in the request"
                    ),
                )
            if request.path == "/v1/responses" and body.get("conversation") is not None:
                return self._fail(
                    session,
                    dialect,
                    TaskError(
                        "@intercept requires stateless Responses requests; conversation "
                        "state would retain the provider's original response"
                    ),
                )
            if request.path == "/v1/chat/completions" and body.get("n", 1) != 1:
                return self._fail(
                    session,
                    dialect,
                    TaskError(
                        "@intercept requires exactly one Chat Completions choice"
                    ),
                )
        logger.debug(
            "intercept %s: id=%s stream=%s",
            request.path,
            session.trace.id,
            dialect.streaming(body),
        )
        # The proxy preserves native JSON fields except model + sampling. `prompt` is only the
        # dialect's typed view for building the trace; the renderer re-derives its own from `body`.
        # A user simulator extends both each turn (`dialect.extend` for wire, `prompt` for trace).
        prompt: Messages
        # `tools` is recorded onto the trace only when a turn commits (below / in `_stream`):
        # the request is ground truth for what the model saw, but a refused or failed request
        # was never seen at all.
        prompt, tools = dialect.parse_request(body)
        # Cache the opening so retries do not advance the simulator twice.
        if (
            session.user is not None
            and session.trace.task.data.prompt is None
            and session.trace.num_turns == 0
        ):
            if session.opening is None:
                session.opening = await session.user("")
            body = dialect.extend(body, None, session.opening)
            prompt = [*prompt, *session.opening]
            # If the simulator ended at the open (its task's `@stop` now fires), the loop's
            # `refused()` below halts the harness before any model call — no special-casing here.
        if session.intercepts:
            replacements: dict[str, MessageContent] = {}
            try:
                for message in graph.prepare_turn(session.trace, prompt).tail:
                    if not isinstance(message, ToolMessage):
                        continue
                    replacement = await session.intercept(message)
                    if replacement is not None:
                        assert isinstance(replacement, ToolMessage)
                        replacements[message.tool_call_id] = replacement.content
            except RolloutError as e:
                return self._fail(session, dialect, e)
            if replacements:
                body = dialect.rewrite_tool_results(body, replacements)
                prompt, tools = dialect.parse_request(body)
                logger.debug(
                    "tool results intercepted: id=%s count=%d",
                    session.trace.id,
                    len(replacements),
                )
        if dialect.streaming(body):
            return await self._stream(request, session, dialect, body, prompt, tools)
        headers = request.headers.copy()
        # A user simulator turns one program request into a multi-turn exchange: after each
        # model turn the simulator's reply is injected as a user turn and the model is
        # re-prompted, so a whole game plays out here and only the final assistant message
        # returns to the (simulator-unaware) program. Without a simulator the loop runs once.
        completion: dict | None = (
            None  # the latest turn's response, returned to the program
        )
        while True:
            try:
                refused = await session.refused()
            except RolloutError as e:
                return self._fail(session, dialect, e)
            except Exception as e:
                return self._fail(
                    session,
                    dialect,
                    TaskError(f"@stop failed: {type(e).__name__}: {e}"),
                )
            if refused is not None:
                # Refuse the first model call to halt the harness; once a simulated
                # conversation is under way, just end it and return the last turn cleanly.
                if completion is None:
                    return web.json_response(
                        dialect.error_body(f"rollout stopped: {refused}"), status=400
                    )
                return _completion_response(completion)
            turn = graph.prepare_turn(session.trace, prompt)
            session.error = None
            try:
                response = await session.ctx.client.get_response(
                    dialect,
                    body,
                    session.ctx.model,
                    session.ctx.sampling,
                    headers=headers,
                    session_id=session.trace.id,
                    turn=turn,
                )
            except OverlongPromptError:
                # An overlong prompt is a budget limit, not a crash: end the rollout cleanly
                # as a truncation — return the last turn if there is one, else refuse to halt
                # the harness (same shape as `refused` above).
                session.trace.stop("context_length")
                logger.debug("prompt too long: id=%s", session.trace.id)
                if completion is None:
                    return web.json_response(
                        dialect.error_body("rollout stopped: context_length"),
                        status=400,
                    )
                return _completion_response(completion)
            except RolloutError as e:
                # Stash the real cause; the rollout re-raises it after the harness returns. Relay
                # the provider's status so the harness SDK retries 5xx/429 and not 4xx.
                session.error = e
                logger.warning(
                    "model call failed: id=%s %s: %s",
                    session.trace.id,
                    type(e).__name__,
                    e,
                )
                return web.json_response(
                    dialect.error_body(str(e)), status=getattr(e, "status_code", 502)
                )
            except Exception as e:  # surface to the program as an API error
                logger.warning(
                    "model call failed: id=%s %s: %s",
                    session.trace.id,
                    type(e).__name__,
                    e,
                )
                return web.json_response(dialect.error_body(str(e)), status=502)
            try:
                replacement = await session.intercept(response.message)
                if replacement is not None:
                    raw = dialect.rewrite_response(response.raw, replacement)
                    response = dialect.parse_response(dialect.validate_response(raw))
                    response.raw = raw
                    if response.id:
                        session.rewritten_response_ids.add(response.id)
                    logger.debug("turn intercepted: id=%s", session.trace.id)
            except RolloutError as e:
                return self._fail(session, dialect, e)
            except (
                Exception
            ) as e:  # a rewrite the dialect cannot serialize is a task bug
                return self._fail(
                    session,
                    dialect,
                    TaskError(f"@intercept failed: {type(e).__name__}: {e}"),
                )
            # `Response.raw` is the full native provider object, or the renderer's synthesized
            # completion object, that the server serializes for the program.
            completion = response.raw
            logger.debug(
                "intercept turn: id=%s tools=%d",
                session.trace.id,
                len(response.message.tool_calls or []),
            )
            turn.commit(response)  # one node per new message;
            # branches fall out of walking the graph (see Trace.branches / verifiers.v1.graph)
            if tools:
                # Persist the advertised tools now that the model actually saw them (see
                # `Trace.tools`). Only a tools-bearing turn updates it (truthy, so an empty
                # parse or a trailing tool-less call never erases the schemas the recorded
                # tool calls need).
                session.trace.tools = tools
            # Hand back to the program when the model wants a tool (the program runs it) or
            # when there's no user simulator to keep the conversation going.
            if response.message.tool_calls or session.user is None:
                return _completion_response(completion)
            try:
                user_messages = await session.user(response.message.content or "")
            except RolloutError as e:
                return self._fail(session, dialect, e)
            except Exception as e:
                return self._fail(
                    session,
                    dialect,
                    UserError(f"user simulator failed: {type(e).__name__}: {e}"),
                )
            # Inject the model turn + the simulator's user turn(s): into the wire request for the
            # next model call (`dialect.extend`, which keeps the model turn verbatim so reasoning
            # survives) and into the typed prompt for the trace. The simulator ends the trajectory
            # through its task's `@stop` (e.g. a `user_finished` flag it set on `self.state`),
            # caught by `refused()` at the top of the next iteration — the interception server holds
            # no opinion about the state's contents.
            body = dialect.extend(body, completion, user_messages)
            prompt = [*prompt, response.message, *user_messages]
            # The simulator changed the payload, so this is a new operation rather than a retry.
            headers.popall("idempotency-key", None)
            headers.popall("x-idempotency-key", None)

    async def _stream(
        self,
        request: web.Request,
        session: RolloutSession,
        dialect: Dialect,
        body: dict,
        prompt: Messages,
        tools: list[Tool] | None = None,
    ) -> web.StreamResponse:
        """A streamed (SSE) model turn: relay the provider's stream through to the program,
        incrementally assembling the response to record on the trace. Single-shot — a streamed
        turn never drives a user simulator (the only client that streams is the eval relay)."""
        try:
            refused = await session.refused()
        except RolloutError as e:
            return self._fail(session, dialect, e)
        except Exception as e:
            return self._fail(
                session, dialect, TaskError(f"@stop failed: {type(e).__name__}: {e}")
            )
        if refused is not None:
            return web.json_response(
                dialect.error_body(f"rollout stopped: {refused}"), status=400
            )
        session.error = None
        try:
            turn = graph.prepare_turn(session.trace, prompt)
            reply = await session.ctx.client.relay(
                dialect,
                body,
                session.ctx.model,
                session.ctx.sampling,
                headers=request.headers,
                session_id=session.trace.id,
            )
        except OverlongPromptError:
            session.trace.stop("context_length")
            logger.debug("prompt too long: id=%s", session.trace.id)
            return web.json_response(
                dialect.error_body("rollout stopped: context_length"), status=400
            )
        except RolloutError as e:
            session.error = e
            logger.warning(
                "model call failed: id=%s %s: %s",
                session.trace.id,
                type(e).__name__,
                e,
            )
            return web.json_response(
                dialect.error_body(str(e)), status=getattr(e, "status_code", 502)
            )
        except Exception as e:  # surface to the program as an API error
            logger.warning("model call failed: id=%s %s", session.trace.id, e)
            return web.json_response(dialect.error_body(str(e)), status=502)
        resp = web.StreamResponse(
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        resp.content_type = reply.content_type.split(";")[0].strip()
        # Parse complete events as they relay. Intercepted streams are spooled until the complete
        # assistant message has been classified; ordinary streams stay on the bounded hot path.
        parser = dialect.stream_parser()
        feed_event = parser.feed
        on_done = parser.on_done
        # One bounded producer avoids per-event tasks; keepalive timeouts only cancel readiness waits.
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=_STREAM_QUEUE_MAXSIZE
        )
        ready = asyncio.Event()
        producer = asyncio.create_task(_queue_chunks(reply.chunks, queue, ready))
        parser_error: Exception | None = None
        event_sizes: list[int] = []
        # SSE events from the turn-ending one onward (the terminal event and any trailing
        # `[DONE]`), withheld until the turn is committed: a client that ends its turn on the
        # terminal event (e.g. codex on `response.completed`) would otherwise reach scoring
        # with the turn still unrecorded.
        deferred: list[bytes] = []
        next_keepalive = asyncio.get_running_loop().time() + _KEEPALIVE_INTERVAL_SECONDS

        def parse_event(chunk: bytes) -> None:
            nonlocal parser_error
            if parser_error is not None:
                return
            try:
                if on_done is not None and is_sse_done_event(chunk):
                    on_done()
                feed_event(chunk)
            except Exception as e:
                parser_error = e

        async def keepalive() -> None:
            nonlocal next_keepalive
            await resp.write(b": keepalive\n\n")
            next_keepalive = (
                asyncio.get_running_loop().time() + _KEEPALIVE_INTERVAL_SECONDS
            )

        async def end_stream() -> None:
            with contextlib.suppress(ConnectionResetError):
                for event in deferred:
                    await resp.write(event)
                await resp.write_eof()

        with (
            SpooledTemporaryFile(max_size=_STREAM_MEMORY_BUFFER)
            if session.intercepts
            else contextlib.nullcontext()
        ) as buffer:
            try:
                await resp.prepare(request)
                while True:
                    try:
                        async with asyncio.timeout(_KEEPALIVE_INTERVAL_SECONDS):
                            await ready.wait()
                    except TimeoutError:
                        await keepalive()
                        continue
                    chunk = queue.get_nowait()
                    if queue.empty():
                        ready.clear()
                    if chunk is None:
                        await producer
                        break
                    if buffer is not None:
                        buffer.write(chunk)
                        event_sizes.append(len(chunk))
                        parse_event(chunk)
                        if asyncio.get_running_loop().time() >= next_keepalive:
                            await keepalive()
                        continue
                    if deferred or dialect.is_terminal_event(chunk):
                        parse_event(chunk)
                        # forwarded after the turn is committed, below
                        deferred.append(chunk)
                        continue
                    await resp.write(chunk)
                    parse_event(chunk)
            except ConnectionResetError:
                return resp
            finally:
                producer.cancel()
                # Let a canceled producer enqueue EOF while unwinding.
                if queue.full():
                    queue.get_nowait()
                await asyncio.gather(producer, return_exceptions=True)
                await reply.close()

            try:
                if parser_error is not None:
                    # A spooled stream was never relayed: the client gets an empty stream
                    # and fails the turn cleanly; nothing commits on either path.
                    raise parser_error
                response = parser.finish()
            except Exception:
                await end_stream()
                raise
            if buffer is not None:
                intercept_task = asyncio.create_task(
                    session.intercept(response.message)
                )
                try:
                    while not intercept_task.done():
                        await asyncio.wait(
                            {intercept_task},
                            timeout=_KEEPALIVE_INTERVAL_SECONDS,
                        )
                        if not intercept_task.done():
                            await keepalive()
                    replacement = await intercept_task
                    if replacement is not None:
                        raw = dialect.rewrite_response(response.raw, replacement)
                        response = dialect.parse_response(
                            dialect.validate_response(raw)
                        )
                        response.raw = raw
                        if response.id:
                            session.rewritten_response_ids.add(response.id)
                        events = dialect.serialize_stream(raw)
                    else:
                        buffer.seek(0)
                        events = (buffer.read(size) for size in event_sizes)
                except ConnectionResetError:
                    intercept_task.cancel()
                    await asyncio.gather(intercept_task, return_exceptions=True)
                    return resp
                except Exception as e:
                    session.error = (
                        e
                        if isinstance(e, RolloutError)
                        else TaskError(f"@intercept failed: {type(e).__name__}: {e}")
                    )
                    logger.warning(
                        "stream interception failed: id=%s %s",
                        session.trace.id,
                        session.error,
                    )
                    await end_stream()
                    return resp

                try:
                    for event in events:
                        if deferred or dialect.is_terminal_event(event):
                            deferred.append(event)
                        else:
                            await resp.write(event)
                except ConnectionResetError:
                    return resp

            try:
                turn.commit(response)
                if tools:  # record only on commit, like the non-streaming path
                    session.trace.tools = tools
                logger.debug("intercept stream turn: id=%s", session.trace.id)
            finally:
                # Release the withheld terminal events only now — after the commit.
                await end_stream()
            return resp

    async def handle_aux(
        self, request: web.Request, dialect: Dialect, route: str
    ) -> web.Response:
        """A non-model-turn side request (an `aux_route`, e.g. Anthropic's `count_tokens`):
        relayed as native JSON, never recorded on the trace."""
        session = self.sessions.get(dialect.secret(request.headers))
        if session is None:
            return web.json_response(dialect.error_body("unauthorized"), status=401)
        logger.debug("intercept aux %s: id=%s", route, session.trace.id)
        try:
            result = await session.ctx.client.relay_aux(
                dialect, route, await request.json(), headers=request.headers
            )
        except RolloutError as e:
            # An aux call isn't a model turn, so don't clobber a pending turn error.
            session.error = session.error or e
            logger.warning(
                "aux call failed: id=%s %s: %s",
                session.trace.id,
                type(e).__name__,
                e,
            )
            return web.json_response(
                dialect.error_body(str(e)), status=getattr(e, "status_code", 502)
            )
        except Exception as e:
            logger.warning("aux call failed: id=%s %s", session.trace.id, e)
            return web.json_response(dialect.error_body(str(e)), status=502)
        return web.json_response(result)

    def _session_for(self, request: web.Request) -> RolloutSession | None:
        """The session a state request belongs to, by its `Authorization: Bearer <secret>` — the
        same per-rollout secret the model routes use (dialect-independent, so parsed directly)."""
        auth = request.headers.get("Authorization", "")
        secret = auth[len("Bearer ") :] if auth.startswith("Bearer ") else ""
        return self.sessions.get(secret)

    async def handle_state_get(self, request: web.Request) -> web.Response:
        """Hand a rollout's tool/user server the current shared `trace.state` (it pulls before each
        `@vf.tool`/`respond` call, so it sees writes from the other servers)."""
        session = self._session_for(request)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        logger.debug("intercept GET /state: id=%s", session.trace.id)
        state = session.trace.state
        return web.Response(
            # TypeAdapter emits UTF-8 bytes directly, avoiding a JSON str copy in aiohttp.
            body=TypeAdapter(type(state)).dump_json(state),
            content_type="application/json",
            charset="utf-8",
        )

    async def handle_task_get(self, request: web.Request) -> web.Response:
        """Hand a launched tool/user server the rollout's task (class ref + JSON) so it can run
        `setup_task` for this rollout — keyed by the same bearer secret as the state channel."""
        session = self._session_for(request)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        logger.debug("intercept GET /task: id=%s", session.trace.id)
        task = session.trace.task.data
        return web.json_response(
            {
                "cls": f"{type(task).__module__}:{type(task).__qualname__}",
                "task": task.model_dump_json(),
            }
        )

    async def handle_state_put(self, request: web.Request) -> web.Response:
        """Replace a rollout's shared `trace.state` with a server's pushed copy (validated into the
        trace's `State` type). Last write wins per call. A task ends the trajectory from state via
        its own `@stop` (run in `RolloutSession.refused` before each model call)."""
        session = self._session_for(request)
        if session is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        logger.debug("intercept PUT /state: id=%s", session.trace.id)
        state_cls = type(session.trace.state)
        raw = await request.read()
        try:
            new_state = state_cls.model_validate_json(raw)
        except ValidationError as e:
            # Reject malformed, over-nested, or mismatched state before it enters the shared channel.
            logger.warning("state PUT rejected: id=%s %s", session.trace.id, e)
            return web.json_response(
                {"error": f"invalid state PUT for {state_cls.__name__}: {e}"},
                status=400,
            )
        session.trace.state = new_state
        return web.json_response({"ok": True})
