# Makes `gateway` an explicit package so `python -m gateway.seed_db` and
# `uvicorn gateway.main:app` resolve identically regardless of how Python's
# namespace-package discovery happens to be configured.
