FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install system dependencies needed for heavy Python packages
RUN apt-get update && \
	apt-get install -y --no-install-recommends build-essential git libopenblas-dev libsndfile1 && \
	rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
	pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 7860

# Run Streamlit with XSRF protection disabled (required for HF Spaces file uploads)
# Use the PORT environment variable set by Spaces to ensure correct binding
CMD ["bash", "-lc", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableXsrfProtection=false"]
