# observability/prometheus/

`prometheus.yml` scrapes vLLM's `/metrics` **directly** — no sidecar, no translation layer — and the gateway's `/metrics` as a separate job, so a failure in one cannot hide the other.

`alerts.yml` — ten rules chosen to *predict* failure rather than report it: prefix cache hit rate collapsing, KV cache filling, queue depth rising, billing writes failing. The SLO breach itself is deliberately last. Thermal throttling is not alertable here and the file says why.
