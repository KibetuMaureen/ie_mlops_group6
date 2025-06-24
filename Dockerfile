FROM python:3.10-slim

# Set environment variables to avoid Poetry's virtualenv and cache prompts
ENV POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry'

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy only dependency declarations first (for Docker layer caching)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . /app

# Set default command
CMD ["python", "src/main.py"]

