FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /opt/ml/code /opt/ml/model

# Copy requirements file
COPY requirements.txt /opt/ml/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

# Set working directory
WORKDIR /opt/ml/code

# Copy your inference script
COPY inference.py /opt/ml/code/inference.py

# Copy your trained model (make sure model.joblib exists in build context)
COPY model.joblib /opt/ml/model/model.joblib

# Set entrypoint
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]