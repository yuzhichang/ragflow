#!/usr/bin/env python3
"""Download the Codex app-server binary for the codex Docker image.

The image only COPIES this binary (see Dockerfile) and downloads nothing at build time,
so the build stays offline and reproducible, and a slow/filtered network is dealt with
here once instead of inside every build. Run this before building the image, and re-run
it when bumping the version so the image and the binary stay in step:

    python3 download_deps.py                      # host arch, pinned version
    python3 download_deps.py --version 0.162.0 --arch amd64 --force
    HTTPS_PROXY=http://proxy:port python3 download_deps.py    # or --proxy

Only the Python standard library is used.
"""

import argparse
import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request

# Pinned release. Keep in step with CODEX_VERSION in ../.env.
CODEX_VERSION = "0.162.0"

# GitHub release assets, keyed by Docker build arch:
#   arch -> (release triple, sha256 of the .tar.gz)
# The same digests are pinned in the Dockerfile's CHECKSUMS comment; keep them in step.
ASSETS = {
    "amd64": (
        "x86_64-unknown-linux-musl",
        "629ef675191921f687ed1e016ba6d3dfc8c171c62fa506e3082a9db32817dc3e",
    ),
    "arm64": (
        "aarch64-unknown-linux-musl",
        "9cb0f864a44aaa1f3dbdec881c04c8e1d48eb9dc8ec2af4a3fadc7ad9b78a469",
    ),
}

DOWNLOAD_TIMEOUT = 120  # seconds, per socket operation
CHUNK = 1 << 20  # 1 MiB
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codex-app-server")


def host_arch():
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    return None


def release_url(version, triple):
    return f"https://github.com/openai/codex/releases/download/rust-v{version}/codex-app-server-{triple}.tar.gz"


def download(url, dest, size_hint=None):
    """Stream url to dest, printing a simple progress line."""
    req = urllib.request.Request(url, headers={"User-Agent": "ragflow-codex-downloader"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or size_hint or 0)
        done = 0
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                sys.stderr.write(f"\r  {done >> 20} / {total >> 20} MiB ({pct}%)")
            else:
                sys.stderr.write(f"\r  {done >> 20} MiB")
            sys.stderr.flush()
    sys.stderr.write("\n")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_member(tarball, member_name, output):
    """Copy a single member out of the tarball to output (no extractall: no path risk)."""
    with tarfile.open(tarball, "r:gz") as tf:
        try:
            member = tf.getmember(member_name)
        except KeyError:
            names = ", ".join(m.name for m in tf.getmembers())
            raise SystemExit(f"error: {member_name!r} not in archive (found: {names})")
        src = tf.extractfile(member)
        if src is None:
            raise SystemExit(f"error: {member_name!r} is not a regular file")
        with src, open(output, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
    os.chmod(output, 0o755)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=CODEX_VERSION, help=f"codex release (default {CODEX_VERSION})")
    parser.add_argument("--arch", choices=sorted(ASSETS), default=host_arch() or "amd64", help="target arch (default: host arch)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="where to write the binary (default: alongside this script)")
    parser.add_argument("--proxy", default=None, help="proxy URL, e.g. http://192.168.1.74:1081 (default: HTTPS_PROXY/https_proxy environment)")
    parser.add_argument("--force", action="store_true", help="re-download even if the output exists")
    args = parser.parse_args()

    triple, expected_sha = ASSETS[args.arch]
    url = release_url(args.version, triple)

    if os.path.exists(args.output) and not args.force:
        print(f"{args.output} already exists; pass --force to re-download.")
        return 0

    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball = os.path.join(tmpdir, "codex-app-server.tar.gz")
        print(f"downloading codex-app-server {args.version} ({args.arch}): {url}")
        download(url, tarball)

        actual_sha = sha256_of(tarball)
        if actual_sha != expected_sha:
            raise SystemExit(f"error: sha256 mismatch for {args.arch}\n  expected {expected_sha}\n  actual   {actual_sha}")
        print(f"  sha256 verified: {actual_sha}")

        extract_member(tarball, f"codex-app-server-{triple}", args.output)

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
