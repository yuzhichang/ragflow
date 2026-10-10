# Codex app-server image (mode 8)

Packages the shared Codex app-server that backs `reasoning` level 8. Upstream publishes no
runtime image, so this directory builds one around the official release binary — a single
static (musl) executable — and publishes it as **`infiniflow/codex:<version>`**, which the
`codex` service in `../docker-compose.yml` pulls.

| File | Purpose |
|---|---|
| `Dockerfile` | alpine image; copies the pre-downloaded binary in and runs it unprivileged. Downloads nothing at build time. |
| `entrypoint.sh` | Writes `CODEX_WS_TOKEN` (env) into the token file the CLI requires, then starts the app-server with capability-token WS auth on `0.0.0.0:8790`. |
| `download_deps.py` | Host-side: fetches + SHA-256-verifies the release binary into `codex-app-server` (git-ignored). |
| `.dockerignore` | Keeps the script and archives out of the build context. |

## Build locally (offline)

```bash
cd docker/codex
python3 download_deps.py                   # add --proxy http://host:port if needed
docker build -t infiniflow/codex:0.162.0 .
```

`COPY` preserves the executable bit, so the Dockerfile deliberately does **not** `chmod` the
binary: a `chmod` layer would re-store the ~216 MiB file and double the image. If the binary
is missing or not executable, the build fails with a clear message — run `download_deps.py`.

## Publish / bump the version

1. Pin the new release in `download_deps.py`: update `CODEX_VERSION` and the per-arch
   `ASSETS` digests. Digests come from the GitHub release metadata:
   ```bash
   curl -fsSL https://api.github.com/repos/openai/codex/releases/tags/rust-v<version> \
     | python3 -c 'import json,sys
for a in json.load(sys.stdin)["assets"]:
    if a["name"].endswith("-unknown-linux-musl.tar.gz") and "codex-app-server-" in a["name"]:
        print(a["name"], a["digest"])'
   ```
2. Download and verify (once per target arch):
   ```bash
   python3 download_deps.py --force                 # host arch
   python3 download_deps.py --force --arch arm64    # if publishing arm64
   ```
3. Build, smoke-test, push:
   ```bash
   docker build -t infiniflow/codex:<version> .
   # smoke test: /readyz -> 200; WS handshake no-auth -> 401, correct bearer -> 101
   docker push infiniflow/codex:<version>
   ```
4. Point the deployment at it: set `CODEX_VERSION` in `../.env` (drives
   `CODEX_IMAGE=infiniflow/codex:${CODEX_VERSION}`) and update the version mentioned in
   `../README.md` if it is pinned there.

## Notes

- The published image is **linux/amd64** only. arm64 hosts fall back to emulation or need a
  local build; publish a multi-arch manifest with `docker buildx` if that matters. Only the
  `-unknown-linux-musl` release assets are used.
- Tags are pinned per version; no `latest` is published.
- The image carries **no** model credentials or proxy settings — the mode-8 engine declares
  the model provider per thread, and the app-server reaches it over the container's network.
- The app-server log may warn that `bubblewrap` is absent. This image ships only the
  app-server binary; mode 8 never executes commands, so the disk sandbox is not exercised.
  OS-level sandboxing would need the full release *package* (which bundles `bwrap`) plus the
  container privileges `bwrap` requires.
