# Use an official Python 3.12.12 image compatible with Raspberry Pi (ARM64)
FROM python:3.12.12-slim-bookworm

# Set the working directory
WORKDIR /app

# Install runtime/build dependencies and keep the image small
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tzdata \
        locales \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# Configure UTF-8 locale for logs/Slack payloads
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TZ=UTC

# Copy pyproject.toml and requirements.txt first
COPY pyproject.toml .
COPY requirements.txt .

# Copy src/ directory (required for editable install to work)
COPY src/ ./src/

# Install uv once and keep it on PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install the package in editable mode (includes all dependencies from pyproject.toml)
RUN uv pip install --system -e .

# Copy remaining application code (tests/, config.yaml)
COPY tests/ ./tests/
COPY config.yaml .

# Set the default command using the installed package
CMD ["python", "-m", "optionscanner.main", "--run-mode", "local", "--config", "config.yaml"]
