"""
Create the gateway database and seed it with demo API keys.

Run once before starting the gateway for the first time:

    python -m gateway.seed_db

Safe to re-run: keys are upserted, and re-running updates a key's name and
budget without resetting its accumulated usage.

The budgets below are chosen against the Stage 2 baseline so that the Stage 4
demonstration is quick rather than theoretical:

  demo-small  -> 2,000 tokens. A single 512-token prompt with a 128-token
                 answer costs ~640 tokens, so this exhausts in 3-4 requests
                 and you can see the 429 without waiting.
  demo-large  -> 5,000,000 tokens. Raised from 500,000 during Stage 11: a
                 single ramp at the protocol's 40-requests-per-level costs
                 roughly 160,000 tokens, and each optimization needs a
                 before/after pair, so 500,000 would have run out mid-comparison
                 - producing 429s that would look like a performance result.

Note that upsert_key updates the budget WITHOUT resetting tokens_used, so
re-running this after a budget change preserves accumulated usage history.
"""

import sys

from gateway import db

DEMO_KEYS = [
    ("dev-key-alpha", "demo-large  (load testing, chat UI)", 5_000_000),
    ("dev-key-beta", "demo-small  (budget exhaustion demo)", 2_000),
    # A SECOND large key, existing only so the fairness test has a distinct
    # identity to misbehave under. Per-key admission limits are keyed on the
    # API key, so an "abusive client" sharing dev-key-alpha with the normal
    # users would be throttled together with them and the experiment would
    # measure nothing. Same budget as alpha so the run cannot end early on a
    # 429 and have it mistaken for load shedding.
    ("dev-key-gamma", "demo-abusive  (fairness test: one key, many streams)", 5_000_000),
]

# A pool of per-user keys for the multi-user simulator.
#
# WHY THIS EXISTS, and it was found by the simulator shedding requests at four
# simulated users: per-key admission limits are keyed on the API key, so a
# population of simulated users all presenting `dev-key-alpha` is not a
# population at all - it is ONE tenant, correctly throttled to its share. The
# capacity measurement would then have been measuring the per-key limit rather
# than the service. Real chat users authenticate separately, so the simulator
# does too.
USER_KEY_COUNT = 48
USER_KEY_PREFIX = "dev-user-"
USER_KEY_BUDGET = 2_000_000


def main() -> None:
    # --reset zeroes accumulated usage as well as applying budgets. Normal
    # seeding deliberately preserves tokens_used, so a container restart does
    # not silently refund everyone - but the budget-exhaustion demo can only be
    # run once against a given key without a way to reset it.
    reset = "--reset" in sys.argv

    db.init_db()
    if reset:
        with db._connect() as conn:  # noqa: SLF001 - deliberate admin action
            conn.execute("UPDATE api_keys SET tokens_used = 0")
        print("  RESET: tokens_used zeroed for all keys\n")

    for key, name, budget in DEMO_KEYS:
        db.upsert_key(key, name, budget)
        print(f"  seeded {key:<16} budget={budget:>8,} tokens   ({name})")

    for i in range(USER_KEY_COUNT):
        key = f"{USER_KEY_PREFIX}{i:02d}"
        db.upsert_key(key, f"sim user {i:02d}", USER_KEY_BUDGET)
    print(f"  seeded {USER_KEY_PREFIX}00..{USER_KEY_COUNT - 1:02d}  "
          f"budget={USER_KEY_BUDGET:>8,} tokens each   (chat_sim population)")

    print(f"\nDatabase ready at: {db.DB_PATH.resolve()}")
    print("Start the gateway with:  uvicorn gateway.main:app --port 8080 --reload")


if __name__ == "__main__":
    main()
