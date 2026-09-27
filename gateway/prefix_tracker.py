"""
A gateway-side estimate of what the engine's prefix cache already holds.

WHY THE GATEWAY NEEDS THIS
--------------------------
Admission is weighted by prompt size, because a long prompt costs a long
prefill. But in a chat service most of a long prompt is NOT new: turn 6 of a
conversation resends turns 1-5, and vLLM serves those from its prefix cache
(measured hit rate 60-90% in every run). Charging the full prompt length made
the gateway treat every turn of a long conversation as a cold 8,000-token
prefill. It charged 4 of 6 slots (the limit then) for work the engine did in
a fraction of a step, and refused ~90% of a long-conversation workload the
GPU could have served.

So the cost is charged on the UNCACHED part: the prompt minus the longest
prefix this gateway has recently forwarded.

HOW
---
Each forwarded request records a running hash over its messages: h1 covers
message 1, h2 covers messages 1-2, and so on. The final hash maps to the
estimated token count of the whole prompt. The next turn of the same
conversation starts with exactly those messages, so walking ITS running hashes
finds the previous turn's entry, and everything up to it is presumed cached.
This mirrors the engine's own prefix cache, which is also a hash chain from the
first block, one level of granularity up.

It is an ESTIMATE, and errs in a known direction. The engine may have evicted
the prefix (LRU, one shared pool), in which case the request costs more than
charged. Two bounds keep that honest: entries expire after `ttl_s`, roughly how
long an idle conversation survives in the pool under load; and the table is
size-capped. Being wrong means admitting a little more prefill than intended
for one request - never a correctness problem, only a latency one, and
bounded by the engine's per-step prefill cap.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict


def _message_bytes(m) -> bytes:
    if not isinstance(m, dict):
        return b"?"
    content = m.get("content")
    if not isinstance(content, str):
        content = repr(content)
    return f"{m.get('role', '')}\x00{content}\x1e".encode("utf-8", "replace")


def running_hashes(messages: list) -> list[bytes]:
    """h[i] identifies messages[0..i] exactly - a chain, like the engine's."""
    out, h = [], hashlib.blake2b(digest_size=16)
    for m in messages:
        h.update(_message_bytes(m))
        out.append(h.copy().digest())
    return out


class PrefixTracker:
    def __init__(self, max_entries: int = 4096, ttl_s: float = 300.0) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._seen: OrderedDict[bytes, tuple[int, float]] = OrderedDict()

    def cached_tokens(self, messages: list, now: float | None = None) -> int:
        """Estimated tokens of the longest recently-forwarded prefix of `messages`.

        The final message is excluded: it is the new question, never cached.
        """
        now = time.monotonic() if now is None else now
        hashes = running_hashes(messages)
        for h in reversed(hashes[:-1]):
            hit = self._seen.get(h)
            if hit is None:
                continue
            tokens, at = hit
            if now - at > self.ttl_s:
                del self._seen[h]
                continue
            return tokens
        return 0

    def remember(self, messages: list, prompt_tokens: int, now: float | None = None) -> None:
        """Record that this exact prompt was forwarded to the engine."""
        if not messages:
            return
        now = time.monotonic() if now is None else now
        h = running_hashes(messages)[-1]
        self._seen[h] = (prompt_tokens, now)
        self._seen.move_to_end(h)
        while len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)
