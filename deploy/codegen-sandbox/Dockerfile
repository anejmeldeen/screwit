# Disposable, locked-down sandbox for LLM-written CadQuery code (SPEC 3.3, X7 brief).
#
# The image carries only Python 3.12 + CadQuery + trimesh (pinned) and the runner. It runs a single
# job read from stdin and writes STL/STEP into /out, which is the ONLY path the caller mounts writable.
# The container is always started by codegen/sandbox.py with: --network none, --read-only, --cap-drop
# ALL, --security-opt no-new-privileges, --user 65534:65534, pids/memory/cpu limits, a tmpfs /tmp, and
# a 30 s wall-clock kill. Nothing else is mounted (no repo, no data dir, no home, no env, no secrets).
#
# Pins match the shared dev venv (cadquery 2.8.0, trimesh 5.2.1) so a program that builds here builds
# in P5's CadService too.
FROM python:3.12-slim-bookworm

# CadQuery's OCP wheel needs a few shared libs for OpenGL/X at import time even headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa \
        libxrender1 \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the dev venv versions (cadquery 2.8.0, trimesh 5.2.1, numpy 2.x) so a program that builds
# here builds in P5's CadService too. cadquery 2.8.0 pulls nlopt, which needs numpy>=2.
RUN pip install --no-cache-dir \
        "cadquery==2.8.0" \
        "trimesh==5.2.1" \
        "numpy==2.5.4"

# The output directory the caller bind-mounts. uid 65534 (nobody) must be able to write it.
RUN mkdir -p /out && chown 65534:65534 /out

COPY runner.py /app/runner.py

# nobody:nogroup. The runtime flags re-assert --user 65534:65534, but set it in the image too so a
# manual `docker run` without the flag still never runs as root.
USER 65534:65534

# uid 65534's home (/nonexistent) is on the read-only root, so point caches at the writable tmpfs
# /tmp. Without this, ezdxf (pulled in by CadQuery) warns on every run about an unwritable cache dir.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app
ENTRYPOINT ["python", "/app/runner.py"]
