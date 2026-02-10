FROM python:3.9-slim

WORKDIR /app

# Install only essential system dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY docker/requirements_docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY src/ ./src/
COPY data/ ./data/
COPY results/ ./results/
COPY notebooks/ ./notebooks/

# Copy entrypoint script
COPY docker/entrypoint_app.sh /entrypoint_app.sh
RUN chmod +x /entrypoint_app.sh

# Expose port for Gradio
EXPOSE 7860

# Use entrypoint script
ENTRYPOINT ["/entrypoint_app.sh"]