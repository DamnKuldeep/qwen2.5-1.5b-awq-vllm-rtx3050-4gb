# Gateway container image - Stage 7.
#
# Build context is the PROJECT ROOT, not gateway/, because the image needs both
# gateway/ and ui/ (the chat page is served by the gateway at /chat, so it must
# ship with it). See deployment/docker/docker-compose.yml.
#
#   docker build -f gateway/Dockerfile -t inference-gateway .

FROM python:3.13-slim

# PYTHONUNBUFFERED so logs appear immediately rather than sitting in a buffer -
# without it, `docker compose logs -f gateway` looks frozen during a request.
# PYTHONDONTWRITEBYTECODE because .pyc files in a container are pure noise.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GATEWAY_DB_PATH=/data/usage.db

WORKDIR /app

# Dependencies are copied and installed BEFORE application code. Docker caches
# each layer, so editing main.py rebuilds only the final layers instead of
# re-running pip install. Copying everything at once would invalidate the pip
# layer on every source change and make each rebuild a minute longer.
COPY gateway/requirements.txt gateway/requirements.txt
RUN pip install --no-cache-dir -r gateway/requirements.txt

# requirements-dev.txt is deliberately NOT installed: the image should not ship
# a test framework it will never run.
COPY gateway/ gateway/
COPY ui/ ui/

# Run as a non-root user. A container process that does not need root should not
# have it - if the app is compromised, this is the difference between an
# attacker holding an unprivileged process and holding the container.
# /data is the SQLite volume mount point and must be writable by that user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8080

# Probes /health (LIVENESS), deliberately not /ready. A container healthcheck
# failing means "this container is broken, restart it" - which is only true if
# the gateway process itself is unwell. Stage 10(c) showed the cost of getting
# this wrong: /health used to return 503 when vLLM died, so Docker marked the
# gateway unhealthy while it was in fact fine and serving honest 502s.
#
# Uses python rather than curl because the slim image has no curl, and adding
# it would mean shipping a network tool in a production image for the sake of a
# health check the runtime can do itself.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=4).status==200 else 1)"

# Seed the database, then serve. PRODUCT_SPEC requires the whole stack to come
# up with ONE command, and a gateway with no API keys is not up in any useful
# sense. seed_db upserts, so restarts do not reset accumulated usage - a key
# exhausted before a restart stays exhausted, which is the correct behaviour.
#
# Written as sh -c rather than a shell script file to avoid CRLF line endings:
# a .sh authored on Windows fails in Linux with an opaque "no such file or
# directory" pointing at the interpreter.
#
# No --reload here. It is a development convenience that watches the filesystem;
# in a container the code cannot change, and the watcher is wasted overhead.
CMD ["sh", "-c", "python -m gateway.seed_db && exec uvicorn gateway.main:app --host 0.0.0.0 --port 8080"]
