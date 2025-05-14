# Combined Dockerfile with MongoDB support
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install required dependencies
RUN pip install python-multipart pymongo streamlit requests

# Copy application files
COPY main.py /app/main.py

# Start Frontend
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
