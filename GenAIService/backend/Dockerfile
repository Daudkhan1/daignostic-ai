# Combined Dockerfile with MongoDB support
FROM continuumio/miniconda3:latest

# Create and activate a Python 3.11 environment
RUN conda create -n chat_env python=3.11 -y && \
    echo "source activate chat_env" >> ~/.bashrc

# Install system-level dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean

# Install required dependencies in the Python 3.11 environment
RUN /bin/bash -c "source activate chat_env && pip install fastapi uvicorn python-multipart pymongo streamlit requests"

# Install dependencies for gemini service
RUN /bin/bash -c "source activate chat_env && pip install google-genai"

# Install dependencies for running maanz_ai_model
RUN /bin/bash -c "source activate chat_env && pip install ultralytics python-decouple torch-geometric==2.4.0"

WORKDIR /app/src

COPY maanz_medical_ai_models-1.0.0-py3-none-any.whl /app/src/maanz_medical_ai_models-1.0.0-py3-none-any.whl
RUN /bin/bash -c "source activate chat_env && pip install maanz_medical_ai_models-1.0.0-py3-none-any.whl --force-reinstall --upgrade"


# Copy application files
COPY src /app/src

# Start MongoDB, Backend, and Frontend
CMD ["conda", "run", "--no-capture-output", "-n", "chat_env", "uvicorn", "app:app", "--host", "0.0.0.0", "--port" ,"8001"]
