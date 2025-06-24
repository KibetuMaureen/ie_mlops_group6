# Use slim Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (helps MLflow, WandB, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire project
COPY . .

# Install Python deps using requirements.txt
RUN pip install --upgrade pip

# Handle environment.yml if it's the only env file
RUN pip install poetry && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes || true

RUN pip install -r requirements.txt

# Expose a port in case you switch to FastAPI later
EXPOSE 8000

# Entry point
CMD ["python", "main.py"]



