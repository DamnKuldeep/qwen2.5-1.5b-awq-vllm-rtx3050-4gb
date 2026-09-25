"""
Gateway — Stage 4: auth, usage tracking, budget enforcement, product dashboard.

Stage 3 established an authenticated streaming proxy. Stage 4 adds the parts
that make it a product: per-key token budgets backed by SQLite, a request
history, and a dashboard answering "how is this being used?" — as distinct
from Grafana (Stage 8), which answers "is the system healthy?".

THE CENTRAL TENSION IN THIS FILE, STATED HONESTLY
--------------------------------------------------
Stage 3 forwarded the request body as raw bytes and never parsed it. That was
Boundary 1's structural defence: a field that is never modelled cannot be
silently dropped.

Budgets need token counts. Those live in the response `usage` object, and for
STREAMING requests vLLM only emits usage if the request carries
`stream_options: {"include_usage": true}`. Most clients will not send it, so
the gateway must add it — which means parsing the body it promised not to parse.

The mitigation is precise: Boundary 1's failure mode is deserializing into a
TYPED model (Pydantic class, dataclass) where undeclared fields vanish on
re-serialization. We parse into an UNTYPED dict. A dict has no schema, so
unknown keys — `response_format` included — survive the round trip intact.

Be clear-eyed anyway: this is weaker than opaque bytes. It now depends on a
property of json.loads/json.dumps rather than on never touching the field.
CONSEQUENCE: Stage 5's contract test is no longer a confirmation of something
structurally unbreakable — it is load-bearing, guarding a real behaviour that
could regress. Stage 10 deliberately breaks it to prove the test bites.

Run:  python -m gateway.seed_db          (once)
      uvicorn gateway.main:app --port 8080 --reload
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from gateway import admission, context, db

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")

# Fallback context window if /v1/models cannot be reached. The real value is
# discovered from the engine at startup - hardcoding what the server is serving
# is the bug that broke the chat UI in Stage 11, and it applies to the window
# just as much as to the model name.
FALLBACK_MAX_MODEL_LEN = int(os.getenv("GATEWAY_MAX_MODEL_LEN", "32768"))

# Reply budget: the default when a client sends no max_tokens, and the ceiling
# when it sends a large one. Both are INJECTED into the request.
#
# WHY THIS IS NOT OPTIONAL. vLLM's OpenAI server, when max_tokens is absent,
# uses `max_model_len - input_length` (entrypoints/utils.py:get_max_tokens).
# With a 32,768-token window that means a client omitting the field asks for
# ~30,000 tokens of generation - and a model that does not emit EOS will
# deliver them, holding an admission slot for seven-plus minutes at ~14 ms per
# token. The stream timeout would eventually cut it, mid-answer. An admission
# limit of six slots whose hold time is unbounded is not a limit.
#
# So the gateway bounds it. The default is generous for chat (1,024 tokens is
# roughly 750 words); the ceiling is what the chat UI's continuation logic
# assumes one round can be. Neither drops a field - Boundary 1 forbids that -
# they add one, the same way stream_options.include_usage is added.
DEFAULT_MAX_TOKENS = int(os.getenv("GATEWAY_DEFAULT_MAX_TOKENS", "1024"))
MAX_OUTPUT_TOKENS = int(os.getenv("GATEWAY_MAX_OUTPUT_TOKENS", "2048"))

# Generation can legitimately run long — the Stage 2 baseline measured P99
# end-to-end latency of 8.96 s at concurrency 8. read=None disables the read
# timeout so a slow-but-healthy generation is never killed mid-stream. Connect
# and write stay bounded: a hang there means vLLM is unreachable, not busy.
TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)

# Hop-by-hop headers are meaningful only for one connection and must not be
# forwarded by a proxy (RFC 9110). Content-Length is stripped too — not
# hop-by-hop, but forwarding the original value while re-framing the body
# produces a length mismatch and a corrupted response.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# The chat UI is served BY the gateway rather than opened as a file, so that it
# is same-origin with the API. A page opened as file:// has origin "null" and
# the browser blocks its response from localhost:8080 under the same-origin
# policy, even though the request itself succeeds. The alternative would be
# adding CORS middleware, which means deciding which origins may call an
# authenticated API - a real security decision, not a config detail. Same-origin
# avoids having to make it at all. Stage 7's container ships this file too.
UI_PATH = Path(__file__).resolve().parent.parent / "ui" / "index.html"


# --------------------------------------------------------------------------
# Lifespan
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # One shared client for the process. Creating an AsyncClient per request
    # would open a fresh TCP connection every time and discard the pool; a
    # long-lived client keeps connections to vLLM warm, removing a handshake
    # from every request's TTFT.
    app.state.client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=TIMEOUT)
    app.state.admission = admission.Admission()
    app.state.stats = admission.Stats()
    app.state.max_model_len = None  # discovered lazily from the engine
    yield
    await app.state.client.aclose()


async def get_max_model_len(app: FastAPI) -> int:
    """Ask the engine what its context window is, once, then cache it.

    Discovered rather than configured, for the same reason the model name is:
    a client that hardcodes what the server is serving breaks on every
    deployment change. `/v1/models` reports `max_model_len` per model, so the
    context policy stays correct across a `--max-model-len` change with no
    gateway edit at all.

    Resolved lazily instead of at startup because the gateway can outlive an
    engine restart, and because a failure here must never prevent the process
    from starting - the fallback keeps the service up with a conservative
    window rather than crash-looping on a dependency.
    """
    cached = getattr(app.state, "max_model_len", None)
    if cached:
        return cached
    try:
        resp = await app.state.client.get("/v1/models", timeout=5.0)
        value = int(resp.json()["data"][0]["max_model_len"])
        if value > 0:
            app.state.max_model_len = value
            return value
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        pass
    return FALLBACK_MAX_MODEL_LEN


app = FastAPI(
    title="Inference Gateway",
    description="Authenticated, budget-limited front door for a self-hosted vLLM server.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def openai_shaped_errors(request: Request, exc: HTTPException):
    """Render errors in OpenAI's envelope so existing clients can parse them.

    PRODUCT_SPEC promises an OpenAI client works by changing only the base URL.
    That promise covers failures too: a client parsing error.message on a 401
    or 429 must keep working.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "insufficient_quota" if exc.status_code == 429 else "invalid_request_error",
                "code": exc.status_code,
            }
        },
        headers=exc.headers or {},
    )


# --------------------------------------------------------------------------
# Auth and budget
# --------------------------------------------------------------------------


async def authenticate(request: Request):
    """Validate the bearer token against the database. Returns the key row."""
    header = request.headers.get("authorization")
    if not header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected: Authorization: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization header. Expected: Authorization: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # to_thread because sqlite3 is blocking: calling it directly in an async
    # handler stalls the event loop for every other in-flight request.
    #
    # FAIL CLOSED. If the database is locked, corrupt, or the disk is full, we
    # cannot know this key's budget - and serving GPU work we cannot bill for
    # is strictly worse than refusing it. A billing system that fails open is
    # how a metered service gives away its capacity during an incident, which
    # is exactly when it can least afford to. 503 rather than 500, because the
    # condition is transient and Retry-After is meaningful.
    # The breaker. A read-side failure is not the only way billing breaks - in
    # WAL mode the read keeps working while the write fails - so a run of failed
    # writes closes the gate here, on the read path, where refusing is cheap.
    if BILLING_ERRORS["consecutive"] >= BILLING_FAILURE_THRESHOLD:
        # Half-open: after the cooldown, let one probe through rather than
        # staying closed forever. Without this the breaker latches and the
        # service never recovers even after the database is fixed.
        since = time.time() - BILLING_ERRORS["opened_at"]
        if since < BILLING_RETRY_S:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Usage accounting has failed {BILLING_ERRORS['consecutive']} times "
                    f"consecutively; refusing to serve work that cannot be billed. "
                    f"Retrying accounting in {BILLING_RETRY_S - since:.0f}s."
                ),
                headers={"Retry-After": str(int(BILLING_RETRY_S - since) + 1)},
            )
        # Probe window: reset the clock so only one request slips through per
        # cooldown, and let this one proceed to find out whether writes work.
        BILLING_ERRORS["opened_at"] = time.time()

    try:
        row = await asyncio.to_thread(db.get_key, token)
    except Exception as exc:  # sqlite3.Error, OSError on a full disk, etc.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Usage accounting unavailable; refusing to serve unbilled work: {exc}",
            headers={"Retry-After": str(admission.RETRY_AFTER_S)},
        )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return row


async def authenticate_dashboard(request: Request):
    """Auth for /dashboard only - added when this stopped being localhost-only.

    /dashboard shows every key's name, usage and recent request history, so it
    cannot stay open once a tunnel (Tailscale Funnel) puts it on the public
    internet. It deliberately does NOT reuse authenticate(): that function also
    runs the billing circuit-breaker, which exists to protect BILLED GPU work,
    not a read-only page. A ?key= query param is accepted alongside the usual
    Authorization header because a browser address bar cannot set headers - the
    header stays the right way for scripts/curl, the query param is what makes
    typing the URL directly work for a human.
    """
    token = None
    header = request.headers.get("authorization")
    if header:
        scheme, _, header_token = header.partition(" ")
        if scheme.lower() == "bearer" and header_token:
            token = header_token
    if not token:
        token = request.query_params.get("key")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Provide an API key: ?key=<api_key> in the URL, or an Authorization: Bearer header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    row = await asyncio.to_thread(db.get_key, token)
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    return row


async def enforce_budget(key_row, started: float) -> None:
    """Reject the request if the key has already spent its budget.

    Checked BEFORE the request, deducted AFTER it completes, because a
    request's token count cannot be known before it is generated. This permits
    bounded overshoot: an in-flight request can push a key past its limit.

    The stricter alternative is reserving max_tokens upfront and refunding the
    unused portion, which never overshoots but rejects requests that would
    have fit. For a v1 with per-key budgets, accepting bounded overshoot is the
    better trade, and matches what most commercial APIs do.

    The refusal is RECORDED before it is raised. A dashboard that shows only
    served requests hides exactly the signal an operator needs — "this key is
    hitting its limit repeatedly" is the reason someone opens the page.
    """
    if key_row["tokens_used"] >= key_row["token_budget"]:
        await record(key_row["key"], None, None, 429, False, started)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Token budget exhausted for key '{key_row['name']}': "
                f"{key_row['tokens_used']:,} of {key_row['token_budget']:,} tokens used."
            ),
            headers={
                "X-Budget-Limit": str(key_row["token_budget"]),
                "X-Budget-Used": str(key_row["tokens_used"]),
                "X-Budget-Remaining": "0",
            },
        )


def budget_headers(key_row) -> dict[str, str]:
    remaining = max(0, key_row["token_budget"] - key_row["tokens_used"])
    return {
        "X-Budget-Limit": str(key_row["token_budget"]),
        "X-Budget-Used": str(key_row["tokens_used"]),
        "X-Budget-Remaining": str(remaining),
    }


# --------------------------------------------------------------------------
# Usage extraction
# --------------------------------------------------------------------------


def extract_usage_from_sse(event: bytes) -> dict | None:
    """Pull the usage object out of one server-sent-event block, if present.

    An SSE event is one or more lines; the payload lines start with 'data: '.
    Intermediate chunks carry "usage": null, and the terminator is the literal
    'data: [DONE]'. Only the final usage chunk has a populated object.
    """
    for line in event.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(obj, dict) and obj.get("usage"):
            return obj["usage"]
    return None


# Billing health, and the circuit breaker that makes "fail closed" true.
#
# THE FINDING THIS EXISTS FOR. Worst case #12 expected a locked ledger to
# produce a 503. It produced a 200. The reason is that the database runs in WAL
# mode, where **readers are never blocked by a writer** - so the budget CHECK
# (a read) kept succeeding while the budget WRITE failed, and the service
# happily served GPU work it could not account for. That is failing OPEN on
# billing, which is exactly what was not wanted.
#
# A read-side check cannot detect this, because the read works. The only
# evidence is a run of failed WRITES, so the breaker watches those: after
# BILLING_FAILURE_THRESHOLD consecutive failures, new requests are refused
# until one succeeds. Some requests are necessarily served unbilled before the
# breaker trips - you only learn the write failed after doing the work - but the
# loss is bounded at a handful rather than "everything until someone notices".
#
# AND IT NEEDS A HALF-OPEN STATE, which the first version did not have.
# Version one reset `consecutive` only on a successful write - but once open,
# the breaker refused every request at the auth check, so no write could ever
# be attempted, so it could never reset. It latched permanently: restoring the
# database left the service still returning 503 forever. The failure matrix
# caught it immediately, which is the argument for running these tests against
# the fix and not only against the original bug.
#
# So after BILLING_RETRY_S the breaker goes HALF-OPEN and lets one request
# through as a probe. If its write succeeds the breaker closes; if it fails,
# the clock restarts and it stays open.
BILLING_ERRORS = {"count": 0, "consecutive": 0, "opened_at": 0.0}
BILLING_FAILURE_THRESHOLD = int(os.getenv("GATEWAY_BILLING_FAILURE_THRESHOLD", "3"))
BILLING_RETRY_S = float(os.getenv("GATEWAY_BILLING_RETRY_S", "10"))


async def record(
    api_key: str, model: str | None, usage: dict | None,
    status_code: int, streamed: bool, started: float,
) -> None:
    """Write one request to the ledger.

    Runs inside a `finally`, which is why the exception handling is not
    optional: if this raised there, the generator would die mid-stream and the
    client would see a truncated response caused purely by a bookkeeping
    failure. The request has already consumed GPU time at that point, so the
    correct behaviour is to deliver it and count the accounting miss.

    Note the asymmetry with `authenticate`, and it is deliberate: the budget
    CHECK fails closed (refuse new work we cannot bill), while the budget WRITE
    fails soft (never damage a response that already succeeded). Together they
    bound the damage from a database outage to "some requests went unbilled",
    with a counter saying exactly how many.
    """
    usage = usage or {}
    try:
        await asyncio.to_thread(
            db.record_request,
            api_key,
            model,
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
            int(usage.get("total_tokens", 0) or 0),
            status_code,
            streamed,
            int((time.perf_counter() - started) * 1000),
        )
        BILLING_ERRORS["consecutive"] = 0
    except Exception:
        BILLING_ERRORS["count"] += 1
        BILLING_ERRORS["consecutive"] += 1
        if BILLING_ERRORS["consecutive"] == BILLING_FAILURE_THRESHOLD:
            BILLING_ERRORS["opened_at"] = time.time()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


async def _upstream_status(request: Request) -> tuple[bool, dict]:
    client: httpx.AsyncClient = request.app.state.client
    try:
        upstream = await client.get("/health", timeout=5.0)
        ok = upstream.status_code == 200
        return ok, {
            "gateway": "ok",
            "vllm": "ok" if ok else f"unhealthy (HTTP {upstream.status_code})",
            "vllm_url": VLLM_BASE_URL,
        }
    except httpx.HTTPError as exc:
        return False, {
            "gateway": "ok",
            "vllm": "unreachable",
            "vllm_url": VLLM_BASE_URL,
            "error": str(exc),
        }


@app.get("/health")
async def health(request: Request):
    """LIVENESS. Returns 200 whenever this process is running.

    Upstream status is included as INFORMATION but does not change the status
    code - and that distinction was learned by breaking it. In Stage 10(c) this
    endpoint returned 503 when vLLM was killed, so the container healthcheck
    failed and Docker marked the GATEWAY unhealthy. The gateway was fine; only
    something else had died.

    That matters because of what a failing liveness probe MEANS to an
    orchestrator: "restart this container". Restarting a healthy gateway helps
    nothing, and it drops every in-flight request that was about to succeed the
    moment the engine came back. Liveness must answer "is this process alive?",
    not "is everything this process depends on alive?".

    Use /ready for the latter.
    """
    _, body = await _upstream_status(request)
    return body


@app.get("/ready")
async def ready(request: Request):
    """READINESS. 503 when the gateway cannot currently serve traffic.

    This is the probe a load balancer should use to decide whether to route
    requests here, and the one Kubernetes should use as its readinessProbe in
    Stage 12. Failing readiness removes the pod from service without killing
    it, so it rejoins automatically once the upstream recovers - which is
    exactly the behaviour observed in Stage 10(c), where the gateway recovered
    on its own with no restart.
    """
    ok, body = await _upstream_status(request)
    if not ok:
        return JSONResponse(status_code=503, content=body)
    return body


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    started = time.perf_counter()
    stats: admission.Stats = request.app.state.stats
    adm: admission.Admission = request.app.state.admission

    key_row = await authenticate(request)
    await enforce_budget(key_row, started)

    raw_body = await request.body()

    # Parse into an UNTYPED dict — see the module docstring. No schema means no
    # field can be dropped. If the body is not valid JSON we forward it
    # untouched and let vLLM produce the error, rather than inventing our own.
    try:
        payload = json.loads(raw_body)
        parsed = isinstance(payload, dict)
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload, parsed = None, False

    is_stream = bool(parsed and payload.get("stream"))
    model = payload.get("model") if parsed else None

    # ---- Context-window policy (see gateway/context.py) --------------------
    # Applied BEFORE admission, so a request that cannot fit is reshaped rather
    # than occupying a slot and then being refused by the engine.
    trimmed_count = 0
    est_prompt_tokens = 0
    clamped_from = 0
    if parsed and isinstance(payload.get("messages"), list):
        window = await get_max_model_len(request.app)
        # Bound the reply. Absent -> inject the default; over the ceiling ->
        # clamp and say so in a header. Whichever field the client used is
        # the one written back, so an OpenAI-style client sees its own field.
        field = "max_completion_tokens" if payload.get("max_completion_tokens") else "max_tokens"
        try:
            requested = int(payload.get(field) or 0)
        except (TypeError, ValueError):
            requested = 0
        if requested <= 0:
            reply_budget = DEFAULT_MAX_TOKENS
            payload[field] = reply_budget
        elif requested > MAX_OUTPUT_TOKENS:
            reply_budget = MAX_OUTPUT_TOKENS
            payload[field] = reply_budget
            clamped_from = requested
        else:
            reply_budget = requested
        # Refuse what cannot fit, before it costs a slot. Trimming preserves
        # system messages and the current question by policy, so a single
        # enormous message survives it untouched and would otherwise reach the
        # engine only to come back as a 400 - having already occupied one of
        # six slots. 413 is the honest status: the entity is too large, and no
        # amount of retrying changes that.
        oversized = context.definitely_exceeds_window(
            payload["messages"], window, reply_budget
        )
        if oversized is not None:
            stats.record_status(413)
            await record(key_row["key"], model, None, 413, is_stream, started)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"This single message is too large for the model's context "
                    f"window: at least ~{oversized:,} prompt tokens against a "
                    f"{window:,}-token window, with {reply_budget:,} reserved for "
                    f"the reply. Older turns are trimmed automatically, but the "
                    f"system prompt and your current message are never dropped - "
                    f"so shorten this message."
                ),
                headers=budget_headers(key_row),
            )

        kept, trimmed_count, est_prompt_tokens = context.fit_messages(
            payload["messages"], window, reply_budget
        )
        if trimmed_count:
            payload["messages"] = kept
            stats.context_trimmed_total += 1
            stats.context_trimmed_messages += trimmed_count

    if parsed and is_stream:
        # vLLM emits a final usage chunk only when asked. Without this, every
        # streamed request would bill zero tokens and budgets would never
        # apply to the chat UI — the main way a person uses this product.
        opts = payload.get("stream_options")
        payload["stream_options"] = {**opts, "include_usage": True} if isinstance(opts, dict) else {"include_usage": True}

    # NOTE (Stage 10b): a field allowlist was deliberately inserted here to
    # reproduce the Boundary 1 failure, and the contract suite caught it with
    # 14 failures - `response_format` plus 12 other silently-dropped fields.
    # It has been removed. Do not reintroduce field filtering here: the whole
    # point of the untyped-dict round trip is that no allowlist exists.
    body = json.dumps(payload).encode() if parsed else raw_body

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() != "authorization"
    }
    headers["content-type"] = "application/json"

    # ---- Admission control -------------------------------------------------
    # The gateway, not the engine, is the limiter. --max-num-seqs is set well
    # above this on purpose (32 against 6 here) so the engine's own cap can
    # never silently shape traffic the gateway believes it is controlling.
    # Weighted by predicted cost, not counted as one unit. The estimate comes
    # from the context policy, which has already walked the messages - so this
    # is free, and it is the same number the trim decision used.
    cost = adm.cost_of(est_prompt_tokens)
    try:
        queued_s = await adm.acquire(key_row["key"], cost)
    except admission.Rejected as rej:
        stats.record_status(503)
        await record(key_row["key"], model, None, 503, is_stream, started)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=rej.detail,
            headers={
                "Retry-After": str(admission.RETRY_AFTER_S),
                "X-Shed-Reason": rej.reason,
                **budget_headers(key_row),
            },
        )

    # release() must run exactly once, on every path: upstream failure, normal
    # completion, client disconnect, or stream timeout. A leaked slot is
    # permanent - the service would degrade one slot at a time until it
    # rejected everything, and nothing in the logs would say why.
    released = False

    async def release_slot() -> None:
        nonlocal released
        if not released:
            released = True
            await adm.release(key_row["key"], cost)

    client: httpx.AsyncClient = request.app.state.client
    try:
        upstream_req = client.build_request(
            "POST", "/v1/chat/completions", content=body, headers=headers
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        await release_slot()
        stats.upstream_errors_total += 1
        stats.record_status(502)
        await record(key_row["key"], model, None, 502, is_stream, started)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream inference server error: {exc}",
        )

    resp_headers = {
        k: v for k, v in upstream_resp.headers.items() if k.lower() not in HOP_BY_HOP
    }
    resp_headers.update(budget_headers(key_row))
    resp_headers["X-Queue-Wait-Ms"] = str(int(queued_s * 1000))
    resp_headers["X-Admission-Cost"] = str(cost)
    if trimmed_count:
        # Surfaced so a client can tell the difference between "the model
        # forgot" and "the gateway dropped the oldest turns to make it fit".
        resp_headers["X-Context-Trimmed-Messages"] = str(trimmed_count)
    if clamped_from:
        resp_headers["X-Max-Tokens-Clamped-From"] = str(clamped_from)

    # ---- Non-streaming: read fully, take usage from the response body -----
    if not is_stream:
        try:
            payload_bytes = await upstream_resp.aread()
        finally:
            await upstream_resp.aclose()
            await release_slot()
        usage = None
        try:
            usage = json.loads(payload_bytes).get("usage")
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            pass
        stats.record_status(upstream_resp.status_code)
        await record(key_row["key"], model, usage, upstream_resp.status_code, False, started)
        return StreamingResponse(
            iter([payload_bytes]),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )

    # ---- Streaming: forward bytes unchanged while watching for usage ------
    async def tee():
        """Yield every byte untouched; sniff the usage chunk on the way past.

        The `finally` block is what makes this Boundary-4 safe: if the client
        disconnects mid-stream, or vLLM dies, or the engine preempts the
        sequence, the generator is closed and the request is STILL recorded.
        The gateway's own accounting therefore survives an engine-side failure
        rather than losing the request silently.

        It is also what releases the admission slot. Note the ordering: the
        slot is held for the WHOLE stream, not just until headers arrive,
        because the engine is still decoding that sequence the entire time.
        Releasing at header time would let the gateway admit unbounded
        concurrent generations while believing it had capped them at six.
        """
        buffer = b""
        usage = None
        first_byte_at = None
        deadline = time.perf_counter() + admission.STREAM_TIMEOUT_S
        try:
            async for chunk in upstream_resp.aiter_raw():
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                    stats.record_ttft(first_byte_at - started)
                yield chunk
                buffer += chunk
                # Split on SSE event boundaries so the buffer stays small
                # rather than accumulating the whole response in memory.
                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    found = extract_usage_from_sse(event)
                    if found:
                        usage = found
                # A client that reads one byte per second cannot be allowed to
                # hold a slot forever (worst case #8). The check is here rather
                # than on a timer because backpressure from a slow reader shows
                # up precisely as a slow `yield`.
                if time.perf_counter() > deadline:
                    break
        finally:
            await upstream_resp.aclose()
            await release_slot()
            stats.record_status(upstream_resp.status_code)
            await record(key_row["key"], model, usage, upstream_resp.status_code, True, started)

    return StreamingResponse(tee(), status_code=upstream_resp.status_code, headers=resp_headers)


@app.get("/metrics")
async def metrics(request: Request):
    """Gateway-side Prometheus metrics.

    Deliberately SEPARATE from vLLM's /metrics rather than proxied or merged.
    Boundary 3's failure mode is a monitoring layer that quietly drops the
    engine's own signals; the defence is that Prometheus scrapes the engine
    directly and this endpoint only ever adds facts the engine cannot know -
    queue depth, shed requests, per-key limits, client-observed TTFT.
    """
    adm: admission.Admission = request.app.state.admission
    stats: admission.Stats = request.app.state.stats
    stats.billing_errors = BILLING_ERRORS["count"]
    return PlainTextResponse(
        stats.render(adm),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/admission")
async def admission_state(request: Request):
    """Human-readable admission state. Useful during load tests and in the docs."""
    return request.app.state.admission.snapshot()


@app.get("/v1/models")
async def models(request: Request):
    await authenticate(request)
    client: httpx.AsyncClient = request.app.state.client
    try:
        upstream = await client.get("/v1/models")
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream inference server error: {exc}",
        )
    return JSONResponse(status_code=upstream.status_code, content=upstream.json())


@app.get("/chat")
async def chat_ui():
    """Serve the minimal chat UI, same-origin with the API. See UI_PATH."""
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Chat UI not found at {UI_PATH}")
    # no-cache so edits to the file show up on refresh during development
    return FileResponse(UI_PATH, media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/dashboard")
async def dashboard(request: Request):
    """The PRODUCT usage dashboard — deliberately not Grafana.

    This answers "how is the product being used?": who is calling, how many
    tokens each key has consumed, what budget remains, what was requested
    recently. Grafana (Stage 8) answers a different question — "is the system
    healthy?" — from the engine's own metrics.

    Keeping them separate, and visibly built differently, is the point.
    Conflating infrastructure health with product usage is a real and common
    mistake; two dashboards make the distinction impossible to lose.

    Authenticated as of the Tailscale Funnel change: this used to say
    "unauthenticated in v1, acceptable because it binds to localhost" - that
    stopped being true the moment the gateway got a public URL. See
    authenticate_dashboard(). /metrics and /admission were deliberately left
    open: Prometheus and the load-testing scripts poll them with no auth
    header, and neither leaks anything more sensitive than counters.
    """
    await authenticate_dashboard(request)
    keys = await asyncio.to_thread(db.list_keys)
    history = await asyncio.to_thread(db.recent_requests, 50)
    summary = await asyncio.to_thread(db.totals)
    return TEMPLATES.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"keys": keys, "history": history, "summary": summary,
                 "vllm_url": VLLM_BASE_URL},
    )
