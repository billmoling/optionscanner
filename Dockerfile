# Use an official Python 3.12.12 image compatible with Raspberry Pi (ARM64)
FROM python:3.12.12-slim-bookworm

# Set the working directory
WORKDIR /app

# Install uv (using the official method)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with uv
RUN uv pip install --system -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the default command (we will override this in docker-compose.yml)
CMD ["python", "main.py", "--run-mode", "docker-scheduled", "--config", "config.yaml"]
