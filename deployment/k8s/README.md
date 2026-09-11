# deployment/k8s/

Not built. The plan was a vLLM Deployment with a GPU resource request and a long `startupProbe`, a gateway Deployment with `livenessProbe: /health` and `readinessProbe: /ready`, and a delete-the-pod recovery measurement.

The property this would demonstrate — the gateway staying Running but NotReady while its engine is down, rather than being restarted — has been measured under Compose instead (`docs/FAILURE_MATRIX.md`, case 5: `/health` 200 throughout, `/ready` 503→200, recovery 83 s). The pod-level version is listed as remaining work in `PROGRESS_LOG.md`.
