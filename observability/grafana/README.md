# observability/grafana/

Provisioned from files, not configured in the UI, so the dashboards ship with the repo.

| Dashboard | What it shows |
| --- | --- |
| `dashboards/vllm-engine.json` | The engine's own metrics, led by KV cache usage and prefix cache hit rate — the two a monitoring wrapper is most likely to drop |
| `dashboards/gateway-capacity.json` | Admission control: client-observed TTFT against the SLO line, admitted vs shed, queue depth, shed reasons, context trimming |
