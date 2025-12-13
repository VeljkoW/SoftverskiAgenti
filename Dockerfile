FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for matplotlib (lighter)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create plots and logs directories  
RUN mkdir -p /app/plots /app/logs

# Expose ports for actor communication
EXPOSE 1900 1901

# Docker uses docker_entry.py, main.py is for local testing
CMD ["python", "docker_entry.py"]
