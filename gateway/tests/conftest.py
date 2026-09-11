"""
Shared pytest fixtures for the gateway tests.

The central fixture is `gateway`, which runs the real FastAPI app against a
STUB upstream instead of vLLM. That stub records the exact request body the
gateway would have sent to vLLM, which is what makes Boundary 1 testable:
we can assert on what crossed the boundary, not merely that the call returned
200. See test_structured_output.py for why that distinction is the whole point.

The stub also means these tests need no GPU, no running vLLM, and finish in
milliseconds - so they can run on every change rather than only when the full
stack happens to be up.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import db, main

TEST_KEY = "test-key-contract"

# A minimal but structurally valid OpenAI chat completion. The gateway reads
# `usage` from this to record token counts, so it must be present.
CANNED_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"ok": true}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """Yield (client, captured) with the gateway wired to a recording stub.

    `captured` is a dict that the stub fills in with what it received:
        captured["body"]    - the parsed JSON body sent upstream
        captured["raw"]     - the exact bytes sent upstream
        captured["headers"] - the headers sent upstream

    A fresh temporary database per test keeps tests independent: budgets
    consumed by one test cannot make another fail.
    """
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    db.upsert_key(TEST_KEY, "contract test key", 1_000_000)

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["raw"] = request.content
        captured["headers"] = dict(request.headers)
        try:
            captured["body"] = json.loads(request.content)
        except json.JSONDecodeError:
            captured["body"] = None

        # Streaming requests get an SSE response so the gateway's tee path is
        # exercised, including its usage extraction.
        #
        # The body MUST be an async generator, not bytes. httpx treats a
        # Response built from bytes as already-read content, so the gateway's
        # `aiter_raw()` raises StreamConsumed - the first version of this stub
        # did exactly that and failed both streaming tests.
        #
        # The general lesson: a stub has to reproduce the MODE of the real
        # dependency, not merely its payload. Same bytes delivered the wrong
        # way exercises a different code path than production takes.
        if isinstance(captured["body"], dict) and captured["body"].get("stream"):
            chunks = [
                b'data: {"id":"c","object":"chat.completion.chunk","choices":'
                b'[{"index":0,"delta":{"content":"hi"}}],"usage":null}\n\n',
                b'data: {"id":"c","object":"chat.completion.chunk","choices":[],'
                b'"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
                b"data: [DONE]\n\n",
            ]

            async def sse_body():
                for chunk in chunks:
                    yield chunk

            return httpx.Response(
                200,
                content=sse_body(),
                headers={"content-type": "text/event-stream"},
            )

        return httpx.Response(200, json=CANNED_RESPONSE)

    transport = httpx.MockTransport(handler)

    # TestClient as a context manager runs the app's lifespan, which creates
    # the real httpx client. We replace it afterwards so the app is otherwise
    # entirely unmodified - the code under test is the shipping code.
    with TestClient(main.app) as client:
        main.app.state.client = httpx.AsyncClient(
            transport=transport, base_url="http://stub-upstream"
        )
        yield client, captured


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {TEST_KEY}"}
