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

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install uv once and keep it on PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python dependencies with uv
RUN uv pip install --system -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the default command (we will override this in docker-compose.yml)
CMD ["python", "main.py", "--run-mode", "local", "--config", "config.yaml"]
