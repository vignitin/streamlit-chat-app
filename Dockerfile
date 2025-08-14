FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Note: .env file should be mounted as a volume or passed via docker-compose
# to avoid baking secrets into the image

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit - default to chat app, but can be overridden
CMD ["streamlit", "run", "chat_app.py", "--server.port=8501", "--server.address=0.0.0.0"]