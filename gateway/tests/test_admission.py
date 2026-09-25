"""
Admission control invariants.

These run against `gateway.admission` directly rather than through the app,
because the properties being asserted are about the allocator itself and are
far easier to stress in isolation.

WHY A LEAK TEST EXISTS AT ALL
------------------------------
`admission.py` says, of releasing a slot: "A leaked slot is permanent - the
service would degrade one slot at a time until it rejected everything, and
nothing in the logs would say why." That is exactly the kind of property that
is true when written, silently broken later, and invisible in production until
the service has quietly throttled itself to zero.

The risk is real and specific. Weighted admission replaced a Semaphore with a
counter guarded by a Condition, because a Semaphore cannot be acquired
N-at-a-time atomically. That introduced a window: if `asyncio.wait_for` times
out between the condition being satisfied and `self._used += cost` running,
the slots are taken and never returned.

It turns out to be safe - cancellation is delivered AT the `await`, so the
increment is unreachable unless the wait succeeded - but "it turns out to be
safe" is a property of today's asyncio, not a guarantee anyone wrote down.
This test writes it down.
"""

import asyncio

import pytest

from gateway import admission


def fresh() -> admission.Admission:
    return admission.Admission()


def test_cost_scales_with_prompt_and_is_capped():
    """A long prompt costs more slots, but never the whole pool."""
    cost_of = admission.Admission.cost_of
    assert cost_of(0) == 1
    assert cost_of(admission.LONG_PROMPT_TOKENS) == 1
    assert cost_of(admission.LONG_PROMPT_TOKENS + 1) == 2
    # No single request may take every slot, or short requests could be
    # starved indefinitely by one giant prefill.
    assert cost_of(10**9) == admission.MAX_REQUEST_COST
    assert admission.MAX_REQUEST_COST < admission.MAX_INFLIGHT


def test_per_key_share_shrinks_as_tenants_arrive():
    """Fairness is a share of the pool, not a fixed quota.

    A fixed cap of 3 against 6 slots hands one tenant half the service however
    many others exist - measured at 10/10 normal users over SLO. The share has
    to fall as contenders arrive.
    """
    adm = fresh()
    alone = adm._key_limit("a")
    for i in range(8):
        adm._per_key[f"tenant{i}"] = 1
    crowded = adm._key_limit("a")
    assert crowded < alone
    assert crowded >= 1, "a contending key must always be able to make progress"


@pytest.mark.parametrize("timeout_s,hold_s", [(0.01, 0.01), (0.004, 0.004)])
def test_no_slot_leak_when_timeouts_race_releases(timeout_s, hold_s, monkeypatch):
    """Slots must all come back when timeouts collide with releases.

    The queue timeout is set equal to the hold time on purpose: that is the
    timing where a waiter's deadline and a holder's release land in the same
    scheduler step, which is the only way the acquire path could take slots
    without returning them.
    """
    monkeypatch.setattr(admission, "MAX_INFLIGHT", 2)
    monkeypatch.setattr(admission, "MAX_QUEUE", 50)
    monkeypatch.setattr(admission, "QUEUE_TIMEOUT_S", timeout_s)
    monkeypatch.setattr(admission, "MAX_INFLIGHT_PER_KEY", 2)
    adm = fresh()

    async def worker(i: int) -> None:
        key = f"k{i % 3}"
        try:
            await adm.acquire(key, 1)
        except admission.Rejected:
            return
        await asyncio.sleep(hold_s)
        await adm.release(key, 1)

    async def run() -> None:
        for _ in range(40):
            await asyncio.gather(*(worker(i) for i in range(8)))
        await asyncio.sleep(0.05)

    asyncio.run(run())

    assert adm.slots_used == 0, f"leaked {adm.slots_used} weighted slots"
    assert adm.inflight == 0, f"leaked {adm.inflight} in-flight counts"
    assert adm.waiting == 0, f"leaked {adm.waiting} queue slots"
    assert not adm._per_key, f"per-key residue: {dict(adm._per_key)}"
    assert adm.admitted_total > 0, "test admitted nothing - it proved nothing"


def test_weighted_cost_is_returned_in_full():
    """A request costing 4 slots must return all 4, not 1."""
    async def run() -> None:
        adm = fresh()
        await adm.acquire("k", 4)
        assert adm.slots_used == 4
        await adm.release("k", 4)
        assert adm.slots_used == 0
        assert adm.inflight == 0
    asyncio.run(run())


def test_rejection_before_admission_leaves_no_residue():
    """A key already at its share is refused without consuming anything."""
    async def run() -> None:
        adm = fresh()
        limit = adm._key_limit("k")
        for _ in range(limit):
            await adm.acquire("k", 1)
        with pytest.raises(admission.Rejected) as exc:
            await adm.acquire("k", 1)
        assert exc.value.reason == "per_key_limit"
        # The refusal must not have consumed a queue slot or a weighted slot.
        assert adm.waiting == 0
        assert adm.slots_used == limit
    asyncio.run(run())
