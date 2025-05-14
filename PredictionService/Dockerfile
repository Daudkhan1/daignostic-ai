# -------- Stage 1: Build Environment --------
    FROM continuumio/miniconda3:latest AS builder

    # Set working directory
    WORKDIR /app
    
    # Install system-level dependencies required by OpenCV
    RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean
    
    # Copy only environment.yml to build Conda env
    COPY environment.yml .
    
    # Create conda environment and install dependencies
    RUN conda env create -f environment.yml && \
        conda clean -afy
    
    # Copy project and pip wheel
    COPY . .
    
    # Activate conda env and install your pip wheel
    SHELL ["conda", "run", "-n", "pytorch", "/bin/bash", "-c"]
    RUN pip install --upgrade pip && \
        pip install maanz_medical_ai_models-1.0.0-py3-none-any.whl --force-reinstall --upgrade
    
    # -------- Stage 2: Runtime Environment --------
    FROM continuumio/miniconda3:latest
    
    # Set working directory
    WORKDIR /app
    
    # Install system-level dependencies
    RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean
    
    # Copy Conda environment from builder stage
    COPY --from=builder /opt/conda /opt/conda
    
    # Copy only necessary project files
    COPY --from=builder /app /app
    
    # Activate conda env in shell
    SHELL ["conda", "run", "-n", "pytorch", "/bin/bash", "-c"]
    
    # Expose port
    EXPOSE 8000
    
    # Set working directory to source
    WORKDIR /app/src
    
    # Start FastAPI app
    CMD ["conda", "run", "--no-capture-output", "-n", "pytorch", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    