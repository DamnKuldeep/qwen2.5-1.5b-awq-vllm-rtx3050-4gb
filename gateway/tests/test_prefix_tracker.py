"""
Cache-aware admission: a long conversation is charged for what is NEW.

The failure these guard against was measured, not imagined: charging every
turn of an 8,000-token conversation as a cold 8,000-token prefill cost 4 of 6
admission slots per turn, and the gateway refused ~90% of a workload whose
turns were ~90% prefix-cache hits in the engine.
"""

from gateway.prefix_tracker import PrefixTracker


def convo(n_turns: int, big: str = "notes " * 4000) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "opening " + big}]
    for i in range(n_turns - 1):
        msgs.append({"role": "assistant", "content": f"reply {i}"})
        msgs.append({"role": "user", "content": f"follow-up {i}"})
    return msgs


def test_next_turn_finds_the_previous_prompt():
    t = PrefixTracker()
    turn1 = convo(1)
    t.remember(turn1, 8000, now=0.0)
    turn2 = convo(2)
    assert t.cached_tokens(turn2, now=1.0) == 8000


def test_the_new_question_is_never_treated_as_cached():
    """Resending the identical prompt must not count its own last message."""
    t = PrefixTracker()
    turn1 = convo(1)
    t.remember(turn1, 8000, now=0.0)
    assert t.cached_tokens(turn1, now=1.0) == 0


def test_an_edited_history_is_a_miss():
    """Same shape, different earlier content: the engine's chain breaks at the
    first changed block, and so must this one."""
    t = PrefixTracker()
    t.remember(convo(1), 8000, now=0.0)
    other = convo(2)
    other[1] = {"role": "user", "content": "a different opening"}
    assert t.cached_tokens(other, now=1.0) == 0


def test_entries_expire():
    """An idle conversation is presumed evicted from the engine's pool."""
    t = PrefixTracker(ttl_s=300)
    t.remember(convo(1), 8000, now=0.0)
    assert t.cached_tokens(convo(2), now=301.0) == 0


def test_table_is_bounded():
    t = PrefixTracker(max_entries=2)
    for i in range(5):
        t.remember([{"role": "user", "content": str(i)}], 10, now=float(i))
    assert len(t._seen) == 2


def test_gateway_charges_second_turn_as_one_slot(gateway, auth):
    """End to end: turn 1 of a long conversation costs several slots, turn 2
    costs one, because only its new question is uncached."""
    client, _ = gateway
    turn1 = convo(1, big="notes " * 5000)          # ~10k tokens by the pessimistic estimate
    r1 = client.post("/v1/chat/completions",
                     json={"model": "m", "messages": turn1}, headers=auth)
    turn2 = convo(2, big="notes " * 5000)
    r2 = client.post("/v1/chat/completions",
                     json={"model": "m", "messages": turn2}, headers=auth)
    assert r1.status_code == 200 and r2.status_code == 200
    assert int(r1.headers["X-Admission-Cost"]) > 1
    assert int(r2.headers["X-Admission-Cost"]) == 1
    assert int(r2.headers["X-Cached-Prefix-Estimate"]) > 0
