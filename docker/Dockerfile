FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ../app/ ./app/
COPY ../src/ ./src/
COPY ../data/ ./data/
COPY ../notebooks/ ./notebooks/

# Copy entrypoint script
COPY entrypoint_app.sh /entrypoint_app.sh
RUN chmod +x /entrypoint_app.sh

# Expose port for Gradio
EXPOSE 7860

# Use entrypoint script
ENTRYPOINT ["/entrypoint_app.sh"]