# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install system dependencies + Python dependencies + AWS CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip \
    && pip install --no-cache-dir -r requirements.txt awscli \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Default command to run the app
CMD ["python", "app.py"]
