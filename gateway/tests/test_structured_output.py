"""
Boundary 1 contract tests: does the gateway actually forward what it was given?

WHY THIS FILE EXISTS
--------------------
The failure mode named in the reading is a gateway that silently drops request
fields - `response_format` being the classic casualty - because it deserializes
into a typed model that does not declare them. It is vicious precisely because
NOTHING LOOKS WRONG: HTTP 200, a plausible response, no error anywhere. The only
symptom is that structured output stops being structured.

A test asserting `status_code == 200` sails straight through that failure. So
these tests assert on WHAT CROSSED THE BOUNDARY, not on whether the call
succeeded. That is what makes them contract tests rather than smoke tests.

WHY THIS IS NOW LOAD-BEARING
----------------------------
In Stage 3 the gateway forwarded raw bytes and never parsed the body, so this
property was structurally unbreakable and these tests would have been a
formality. Stage 4 traded that away in order to inject
`stream_options.include_usage` for budget accounting: the body is now parsed
into an untyped dict and re-serialized. Untyped means no schema, so nothing is
dropped - but the guarantee now rests on a property of json.loads/json.dumps
rather than on never touching the field. That is a real, breakable behaviour.
Stage 10 breaks it deliberately to prove these tests bite.

Run:  pytest gateway/tests -v
      pytest gateway/tests -v -m integration   (needs the full stack running)
"""

import json

import pytest

# Contract tests run against a stub upstream, so this value is arbitrary -
# the stub echoes whatever it is sent. The INTEGRATION test resolves the real
# served model from /v1/models instead of using this, because a hardcoded name
# 404s the moment the engine serves something else.
MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"


def resolve_served_model(base_url: str, api_key: str, fallback: str = MODEL) -> str:
    """Ask the gateway which model the engine is actually serving."""
    import httpx as _httpx

    try:
        resp = _httpx.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        return resp.json()["data"][0]["id"]
    except Exception:  # noqa: BLE001 - fall back rather than fail the test setup
        return fallback


def base_request(**extra) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(extra)
    return body


# --------------------------------------------------------------------------
# The core Boundary 1 property
# --------------------------------------------------------------------------


def test_response_format_reaches_upstream(gateway, auth):
    """response_format must arrive at vLLM byte-identical to what was sent.

    This is the specific field the reading names. Asserting equality of the
    whole object - not merely that the key exists - catches a gateway that
    forwards a truncated or rewritten version.
    """
    client, captured = gateway
    sent = {"type": "json_object"}

    resp = client.post(
        "/v1/chat/completions", json=base_request(response_format=sent), headers=auth
    )

    assert resp.status_code == 200
    assert "response_format" in captured["body"], (
        "response_format was DROPPED by the gateway. This is the Boundary 1 "
        "failure: the client asked for structured output and vLLM never heard "
        "about it. The response would still be a valid-looking 200."
    )
    assert captured["body"]["response_format"] == sent


def test_unknown_future_field_survives(gateway, auth):
    """A field the gateway has never heard of must still be forwarded.

    THIS IS THE REAL PROPERTY. Testing `response_format` specifically only
    proves someone remembered that one field. The actual guarantee we want is
    that the gateway has no allowlist at all - so a field added to the OpenAI
    API next year, which no one will remember to update this gateway for,
    passes through untouched.

    A typed-model gateway fails this test while passing every field-specific
    one, which is exactly how the failure survives into production.
    """
    client, captured = gateway
    payload = base_request(
        some_field_invented_in_2027={"nested": ["a", 1, None, True]},
        another_unknown_flag=False,
    )

    resp = client.post("/v1/chat/completions", json=payload, headers=auth)

    assert resp.status_code == 200
    assert captured["body"]["some_field_invented_in_2027"] == {
        "nested": ["a", 1, None, True]
    }
    assert captured["body"]["another_unknown_flag"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.0),
        ("top_p", 0.85),
        ("top_k", 20),
        ("seed", 12345),
        ("stop", ["\n\n", "END"]),
        ("max_tokens", 256),
        ("presence_penalty", 0.5),
        ("frequency_penalty", -0.25),
        ("logit_bias", {"1234": -100}),
        ("n", 2),
        ("user", "customer-42"),
        ("tools", [{"type": "function", "function": {"name": "get_weather"}}]),
        ("tool_choice", "auto"),
        ("guided_json", {"type": "object", "properties": {"x": {"type": "integer"}}}),
    ],
)
def test_request_field_survives(gateway, auth, field, value):
    """Every field a client might send must arrive unchanged.

    Includes vLLM-specific extensions (guided_json) alongside standard OpenAI
    parameters, because a gateway that only models the OpenAI schema would
    silently strip the vendor extensions that make this deployment useful.
    """
    client, captured = gateway
    resp = client.post(
        "/v1/chat/completions", json=base_request(**{field: value}), headers=auth
    )
    assert resp.status_code == 200
    assert captured["body"][field] == value, f"gateway dropped or altered '{field}'"


def test_messages_are_not_reshaped(gateway, auth):
    """Message list structure, roles, ordering and extra keys must be preserved."""
    client, captured = gateway
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi", "name": "alice"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "content": "42", "tool_call_id": "call_1"},
    ]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": messages},
        headers=auth,
    )
    assert resp.status_code == 200
    assert captured["body"]["messages"] == messages


# --------------------------------------------------------------------------
# The one field the gateway is ALLOWED to change, and its limits
# --------------------------------------------------------------------------


def test_stream_options_injected_for_streaming(gateway, auth):
    """The gateway adds include_usage so streamed requests can be billed.

    This is the single deliberate modification the gateway makes, and the
    reason the body is parsed at all. Without it vLLM reports no usage for a
    stream, every streamed request bills zero tokens, and budgets silently do
    not apply to the chat UI - the main way a person uses this product.
    """
    client, captured = gateway
    with client.stream(
        "POST", "/v1/chat/completions", json=base_request(stream=True), headers=auth
    ) as resp:
        resp.read()
    assert captured["body"]["stream_options"] == {"include_usage": True}


def test_client_stream_options_are_merged_not_replaced(gateway, auth):
    """A client's own stream_options keys must survive the injection.

    The obvious implementation - assigning stream_options wholesale - would
    silently discard whatever the client set. That is the Boundary 1 failure
    reappearing inside the one place we permitted ourselves to modify.
    """
    client, captured = gateway
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=base_request(stream=True, stream_options={"some_other_option": "keep-me"}),
        headers=auth,
    ) as resp:
        resp.read()

    assert captured["body"]["stream_options"]["some_other_option"] == "keep-me"
    assert captured["body"]["stream_options"]["include_usage"] is True


def test_missing_max_tokens_is_bounded(gateway, auth):
    """A request with no max_tokens must not reach the engine with none.

    vLLM's OpenAI server defaults an absent max_tokens to the REST OF THE
    CONTEXT WINDOW (max_model_len - input_length). On a 32k window that is
    ~30,000 tokens of generation for a client that simply forgot the field -
    holding one of six admission slots for minutes. The gateway injects a
    bounded default so the admission limit means what it says.
    """
    client, captured = gateway
    body = base_request()
    body.pop("max_tokens", None)
    resp = client.post("/v1/chat/completions", json=body, headers=auth)
    assert resp.status_code == 200
    assert isinstance(captured["body"].get("max_tokens"), int)
    assert 0 < captured["body"]["max_tokens"] <= 2048


def test_oversized_max_tokens_is_clamped_and_reported(gateway, auth):
    """A client asking for 30,000 output tokens gets the ceiling, and is told."""
    client, captured = gateway
    resp = client.post(
        "/v1/chat/completions", json=base_request(max_tokens=30_000), headers=auth
    )
    assert resp.status_code == 200
    assert captured["body"]["max_tokens"] == 2048
    assert resp.headers.get("X-Max-Tokens-Clamped-From") == "30000"


def test_reasonable_max_tokens_is_untouched(gateway, auth):
    """Inside the ceiling the client's value must arrive exactly as sent.

    The bound exists to stop unbounded holds, not to second-guess a client
    that asked for something sane. Altering a value inside the limit would be
    the Boundary 1 failure wearing a different hat.
    """
    client, captured = gateway
    resp = client.post(
        "/v1/chat/completions", json=base_request(max_tokens=777), headers=auth
    )
    assert resp.status_code == 200
    assert captured["body"]["max_tokens"] == 777
    assert "X-Max-Tokens-Clamped-From" not in resp.headers


def test_single_oversized_message_is_refused_with_413(gateway, auth):
    """One enormous message must be refused, not forwarded to the engine.

    The context policy never drops system messages or the current question, so
    a single huge message survives trimming untouched. Before this guard it
    reached vLLM, occupied one of six admission slots, and came back as the
    engine's 400 - and the README claimed overflow was "never a 400".

    413 is the correct status and the gateway is the correct place: no GPU time
    is spent, and no slot is held.
    """
    client, captured = gateway
    huge = "word " * 200_000          # ~1M characters
    resp = client.post(
        "/v1/chat/completions",
        json=base_request(messages=[{"role": "user", "content": huge}]),
        headers=auth,
    )
    assert resp.status_code == 413
    assert "/v1/chat/completions" not in captured.get("paths", []), (
        "oversized request must never reach the engine"
    )
    assert "context window" in resp.json()["error"]["message"]


def test_large_but_fitting_message_is_not_refused(gateway, auth):
    """The refusal must not fire on prompts that would actually have fit.

    Trimming deliberately OVER-estimates tokens so it trims early; refusing
    deliberately UNDER-estimates so it only refuses the certain cases. This
    asserts the asymmetry: a prompt comfortably inside the window is forwarded
    even though the pessimistic estimate used for trimming is much larger.
    """
    client, captured = gateway
    # ~40k characters: well under a 32,768-token window on any tokenisation,
    # but over it under the pessimistic chars/3.0 ratio used for trimming.
    body = base_request(messages=[{"role": "user", "content": "x " * 20_000}])
    resp = client.post("/v1/chat/completions", json=body, headers=auth)
    assert resp.status_code == 200
    assert captured.get("body") is not None


def test_no_stream_options_added_to_non_streaming(gateway, auth):
    """Non-streaming requests must not get stream_options.

    Usage is available directly in a non-streamed response body, so there is
    no reason to inject it. (The gateway DOES bound max_tokens on every
    request, streaming or not - see the max_tokens tests above - because that
    protects the admission limit rather than the billing path.)
    """
    client, captured = gateway
    resp = client.post("/v1/chat/completions", json=base_request(), headers=auth)
    assert resp.status_code == 200
    assert "stream_options" not in captured["body"]


def test_non_json_body_is_forwarded_untouched(gateway, auth):
    """An unparseable body is passed through, not rejected by the gateway.

    The upstream should produce the error. A gateway inventing its own
    validation error would report a failure vLLM might not have produced, and
    would diverge from the OpenAI-compatible behaviour the spec promises.
    """
    client, captured = gateway
    raw = b"this is not json at all"
    resp = client.post(
        "/v1/chat/completions",
        content=raw,
        headers={**auth, "Content-Type": "application/json"},
    )
    assert resp.status_code == 200  # the stub accepts it; vLLM would 400
    assert captured["raw"] == raw


def test_authorization_header_not_leaked_upstream(gateway, auth):
    """Our API key is this gateway's concept and must not reach vLLM.

    Forwarding it would leak a customer credential to a downstream service
    that has no use for it and logs its request headers.
    """
    client, captured = gateway
    client.post("/v1/chat/completions", json=base_request(), headers=auth)
    assert "authorization" not in {k.lower() for k in captured["headers"]}


# --------------------------------------------------------------------------
# Integration: the real stack. Skipped by default (see pytest.ini).
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_json_object_mode_end_to_end():
    """Against real vLLM, json_object mode must return parseable JSON.

    The contract tests above prove the field is FORWARDED. This proves the
    whole path actually WORKS - that vLLM honours it and the client gets JSON.
    Both are needed: forwarding a field to a server that ignores it is not a
    working feature, and a feature that works today can silently stop being
    forwarded tomorrow.

    Requires: vLLM running, gateway running, `python -m gateway.seed_db` done.
    """
    import httpx as _httpx

    served = resolve_served_model("http://localhost:8080", "dev-key-alpha")

    resp = _httpx.post(
        "http://localhost:8080/v1/chat/completions",
        headers={"Authorization": "Bearer dev-key-alpha"},
        json={
            "model": served,
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with keys 'city' and 'country' for Paris.",
                }
            ],
            "temperature": 0,
            "max_tokens": 60,
            "response_format": {"type": "json_object"},
        },
        timeout=120.0,
    )

    assert resp.status_code == 200, resp.text
    content = resp.json()["choices"][0]["message"]["content"]

    # The assertion that matters: parse the BODY. A status-code check would
    # pass even if response_format had been dropped entirely.
    parsed = json.loads(content)
    assert isinstance(parsed, dict), f"expected a JSON object, got: {content!r}"
