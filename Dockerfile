# Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY environment.yml .
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes || true && \
    pip install -r requirements.txt || true

# Copy project files
COPY . .

# Expose port used by FastAPI
EXPOSE 8000

# Start FastAPI app (update path if needed)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
