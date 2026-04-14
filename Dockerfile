FROM python:3.11-slim-bookworm

# System libraries required by Python packages:
#   pycurl     → libcurl4-openssl-dev
#   tables     → libhdf5-dev
#   pyzmq      → libzmq3-dev
#   pycares    → libcares-dev
#   pyopengl   → libgl1 (headless runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libcurl4-openssl-dev \
        libssl-dev \
        libhdf5-dev \
        libffi-dev \
        libzmq3-dev \
        libcares-dev \
        pkg-config \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Bring in uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy lock files first so dep installation is cached independently of source changes
COPY pyproject.toml uv.lock ./

# Install all declared dependencies but not the project package itself
RUN uv sync --frozen --no-install-project

# Copy source and install the project package
COPY metatlas2/ ./metatlas2/
RUN uv sync --frozen

# Put the virtualenv's python on PATH
ENV PATH="/app/.venv/bin:$PATH"

# The image tag is injected by CI at build time and surfaced at runtime
# so the running code can record which image generated the analysis.
ARG IMAGE_TAG=latest
ENV METATLAS2_IMAGE_TAG=${IMAGE_TAG}

ENTRYPOINT ["python", "-m", "metatlas2.run_targeted_analysis"]
CMD ["--help"]
