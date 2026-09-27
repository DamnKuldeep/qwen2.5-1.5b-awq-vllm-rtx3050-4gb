"""
Context-window policy — what happens when a conversation outgrows the model.

THE PROBLEM
-----------
A stateless chat API resends the whole conversation every turn, so context
grows without bound while `--max-model-len` does not. v1 had no policy at all:
the request simply reached vLLM, which refused it with a 400, and the chat UI
showed an error mid-conversation with no way forward. That is a hard failure in
normal use, not an edge case - any long enough conversation reaches it.

THE POLICY: SLIDING WINDOW THAT PRESERVES THE SYSTEM PROMPT
------------------------------------------------------------
Drop whole messages from the OLDEST end until the request fits, always keeping:

  * every `system` message (they carry instructions, and they are the shared
    prefix that makes prefix caching work), and
  * the most recent user message (dropping the question being asked is absurd).

Three alternatives were considered:

  * **Reject with a clear error.** Honest and trivial, but it makes the product
    unusable at exactly the point a user is most invested in the conversation.
  * **Summarise older turns.** Best quality retention, but it costs an extra
    generation per overflow, and on a 1.5B model the summary quality is the
    weakest link. Out of scope here; noted as the upgrade path.
  * **Truncate mid-message.** Cheapest, and produces incoherent context - the
    model sees half a sentence and continues it. Rejected.

THE INTERACTION THAT MATTERS, AND IT IS NOT OBVIOUS
----------------------------------------------------
A sliding window BREAKS THE PREFIX CACHE. vLLM's prefix hash is a chain: block
N incorporates blocks 1..N-1, so a hit needs a contiguous prefix from the very
start. Dropping the oldest turn changes the first block, which invalidates
every block after it. The turn that trims is therefore a guaranteed full-prefill
miss - and on this hardware a miss costs roughly 9x a hit.

Keeping system messages pinned at the front limits the damage: the system
prefix stays byte-identical across all users and all turns, so that portion
stays cached. This is why the policy preserves them by position, not merely by
presence.

TOKEN COUNTING
--------------
Estimated locally rather than by calling vLLM's /tokenize, because a tokenize
round trip sits directly in the TTFT path and TTFT is the SLO. The estimate is
deliberately CONSERVATIVE - it overestimates - so the policy trims slightly
early rather than letting a request through that vLLM will refuse. Over-trimming
costs a little history; under-trimming costs a hard 400 in the user's face.
"""

from __future__ import annotations

# Qwen2.5 averages roughly 4 characters per token on English prose. Dividing by
# 3.0 overestimates by ~33%, which is the safety margin: we would rather trim a
# turn early than emit a request the engine rejects.
CHARS_PER_TOKEN = 3.0

# The OPPOSITE bias, and it exists for the opposite decision.
#
# Trimming uses the pessimistic ratio above, because trimming one turn early is
# cheap. Refusing a request that would actually have fit is a bug the client
# cannot work around, so this optimistic ratio is only a PRE-FILTER: it decides
# whether it is worth asking the engine's real tokenizer, and the refusal itself
# is always made on the exact count (gateway/main.py).
#
# It cannot be the final word. No character ratio is a lower bound for BPE: a
# prompt of common English words measured 8.3 chars/token on Qwen, so ~20,000
# real tokens looked like ~37,000 here and a prompt that fit was refused with
# 413. The failure matrix caught it - case 3's big prompt silently stopped
# being served.
OPTIMISTIC_CHARS_PER_TOKEN = 4.5

# Per-message chat-template overhead (role markers, delimiters). Qwen's template
# adds about 4; 8 is deliberate headroom.
PER_MESSAGE_OVERHEAD = 8

# Template preamble plus the generation prompt suffix.
TEMPLATE_OVERHEAD = 16


def _content_text(content) -> str:
    """Flatten a message's content, which may be a string or a content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                # Text parts carry "text"; image parts have no cheap token
                # estimate, so they are charged a flat, generous constant.
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(" " * int(CHARS_PER_TOKEN * 256))
            else:
                parts.append(str(part))
        return "".join(parts)
    return "" if content is None else str(content)


def estimate_tokens(message: dict) -> int:
    text = _content_text(message.get("content"))
    # Tool calls and names are part of the rendered prompt too.
    for extra in ("name", "tool_call_id"):
        if message.get(extra):
            text += str(message[extra])
    if message.get("tool_calls"):
        text += str(message["tool_calls"])
    return int(len(text) / CHARS_PER_TOKEN) + PER_MESSAGE_OVERHEAD


def estimate_prompt_tokens(messages: list) -> int:
    return TEMPLATE_OVERHEAD + sum(
        estimate_tokens(m) for m in messages if isinstance(m, dict)
    )


def fit_messages(
    messages: list,
    max_model_len: int,
    max_tokens: int,
) -> tuple[list, int, int]:
    """Trim oldest turns until prompt + max_tokens fits the context window.

    Returns (messages, dropped_count, estimated_prompt_tokens).

    The budget reserves `max_tokens` for the reply, because `--max-model-len`
    caps prompt PLUS generation, not the prompt alone. A request that fits on
    arrival but has no room to answer is still a failed request.
    """
    if not isinstance(messages, list) or not messages:
        return messages, 0, 0

    budget = max_model_len - max(0, max_tokens)
    if budget <= 0:
        # max_tokens alone exceeds the window; nothing to trim can fix that, so
        # forward it and let the engine produce the authoritative error.
        return messages, 0, estimate_prompt_tokens(messages)

    total = estimate_prompt_tokens(messages)
    if total <= budget:
        return messages, 0, total

    # Index the messages we are allowed to drop: everything except system
    # messages and the final message (the question being asked right now).
    last_index = len(messages) - 1
    droppable = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") != "system" and i != last_index
    ]

    dropped: set[int] = set()
    for i in droppable:
        if total <= budget:
            break
        total -= estimate_tokens(messages[i])
        dropped.add(i)

    kept = [m for i, m in enumerate(messages) if i not in dropped]
    return kept, len(dropped), total


def undroppable(messages: list) -> list:
    """The messages trimming will never remove: system prompts and the last one.

    The complement of `fit_messages`' droppable set. The oversize check counts
    only these, because whatever trimming can remove is no reason to refuse.
    """
    if not isinstance(messages, list) or not messages:
        return []
    last_index = len(messages) - 1
    return [
        m for i, m in enumerate(messages)
        if isinstance(m, dict) and (m.get("role") == "system" or i == last_index)
    ]


def probably_exceeds_window(
    messages: list,
    max_model_len: int,
    max_tokens: int,
) -> int | None:
    """Optimistic token estimate of the undroppable messages, when it overflows.

    Returns that estimate when it exceeds the window, and None otherwise. A
    non-None result is a reason to COUNT EXACTLY, not a reason to refuse - see
    OPTIMISTIC_CHARS_PER_TOKEN for why no character ratio can be trusted here.

    WHY THIS EXISTS. `fit_messages` cannot drop system messages or the final
    user message - dropping the question being asked is absurd, and the system
    prompt is the shared prefix that makes prefix caching work. So a single
    enormous message survives trimming untouched, reaches the engine, and comes
    back as vLLM's 400 *after* it has occupied an admission slot.

    A measured fuzz of the policy found exactly that: one 333,000-token user
    message, `dropped=0`, straight through. The README claimed context overflow
    was "never a 400"; for multi-turn overflow that was true, and for this case
    it was not.

    Catching it at the gateway turns an engine-side 400 into a 413 that names
    the limit, costs no GPU time, and holds no slot.

    ONLY THE UNDROPPABLE MESSAGES ARE COUNTED - the same set `fit_messages`
    keeps. The first version counted every message, so an ordinary long
    conversation (many turns, each small) was refused with 413 instead of
    being trimmed. The failure matrix caught it: case 4 regressed from
    "200, oldest turns dropped" to "413". Whatever trimming can remove is not
    a reason to refuse.
    """
    kept = undroppable(messages)
    if not kept:
        return None
    chars = sum(len(_content_text(m.get("content"))) for m in kept)
    optimistic = int(chars / OPTIMISTIC_CHARS_PER_TOKEN) + TEMPLATE_OVERHEAD
    budget = max_model_len - max(0, max_tokens)
    return optimistic if optimistic > budget else None
