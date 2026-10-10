# Codex app-server, packaged for the mode-8 engine.
#
# The app-server binary is NOT downloaded during the build: run ./download_deps.py first
# to fetch and SHA-256-verify it into this directory, then build. The image only copies it
# in, so the build is offline and reproducible, and a slow or filtered network is handled
# once by the script instead of on every build.
#
#   python3 download_deps.py            # honours HTTPS_PROXY/https_proxy, or --proxy URL
#   docker compose build codex
#
# The image carries no model credentials: the mode-8 engine declares the model provider
# per thread at run time. Alpine is a natural base — the binary is statically linked
# against musl, and the base already ships a CA bundle the app-server uses for outbound
# model calls.
FROM alpine:3.19

# Run unprivileged, with a writable CODEX_HOME (config/state) and an isolated work dir.
RUN addgroup -g 10001 codex \
    && adduser -D -u 10001 -G codex -h /home/codex codex \
    && mkdir -p /home/codex/.codex /work \
    && chown -R codex:codex /home/codex /work
ENV CODEX_HOME=/home/codex/.codex

COPY entrypoint.sh /usr/local/bin/codex-entrypoint.sh

# Placed here by ./download_deps.py (git-ignored; it sets the executable bit). COPY
# preserves the mode, so there is deliberately NO `RUN chmod` — chmod would re-store the
# whole 200+ MiB binary as a second layer and double the image size. Assert the mode
# instead, so a build after a forgotten download_deps.py fails clearly.
COPY codex-app-server /usr/local/bin/codex-app-server
RUN test -x /usr/local/bin/codex-app-server && test -x /usr/local/bin/codex-entrypoint.sh \
    || { echo "error: codex-app-server/entrypoint.sh not executable — run docker/codex/download_deps.py before building" >&2; exit 1; }

USER codex
WORKDIR /work
EXPOSE 8790
ENTRYPOINT ["/usr/local/bin/codex-entrypoint.sh"]
