FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create data directories as root (before switching user)
RUN mkdir -p /app/knowledge /app/canon /app/mempalace /app/lancedb /app/logs

# Create non-root user for security and give ownership
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

USER app

# Copy requirements first for layer caching
COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app ui/ ./ui/
COPY --chown=app:app tests/ ./tests/

# Set Python path for imports
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "-m", "src.main"]
