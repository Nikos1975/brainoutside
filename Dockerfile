# Brain server image — one image, three roles (web / mcp / worker), the
# command decides. Debian-slim on purpose: the claude-agent-sdk bundles a
# glibc-linked CLI binary, so Alpine/musl is off the table (PLAN.md §7).
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=config.settings.prod

# git: the template image lacked it and the brain IS a git clone (§7).
# openssh-client: clone/pull over the read-only deploy key.
# gosu: entrypoint drops root -> app after fixing /data ownership (A5c).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git openssh-client curl gosu \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; uid 1000 matches the volume-ownership contract.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY requirements.txt .
COPY packages/tiered_openrouter ./packages/tiered_openrouter
RUN pip install --no-cache-dir -r requirements.txt

# Build-time smoke test (grill C1): the SDK imports and its bundled CLI
# answers --version. A broken pin fails the build, not the first request.
RUN python - <<'EOF'
import pathlib, subprocess, sys
import claude_agent_sdk
bundled = pathlib.Path(claude_agent_sdk.__file__).parent / "_bundled"
exe = next((p for p in bundled.iterdir() if p.stem == "claude"), None)
assert exe is not None, f"no bundled CLI under {bundled}"
out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=120)
assert out.returncode == 0, out.stderr
print("claude-agent-sdk OK, bundled CLI:", out.stdout.strip())
EOF

COPY . .

RUN mkdir -p /data && chown -R app:app /app /data
VOLUME ["/data"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]
# Default command = web; compose overrides for mcp / worker.
CMD ["web"]
