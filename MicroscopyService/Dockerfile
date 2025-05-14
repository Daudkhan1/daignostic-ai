FROM debian:stable-slim AS build

# Install system-level dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean

RUN apt-get update \
    && apt-get install -y -q build-essential wget \
    && apt-get clean \
    && apt-get autoremove

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh && /bin/bash /opt/miniconda.sh -b -p /opt/conda

COPY env.yaml /opt/app/
# Install env
RUN /opt/conda/bin/conda env create -n chat_env --file=/opt/app/env.yaml

# Install pip package
WORKDIR /app/src
COPY maanz_medical_ai_models-1.0.0-py3-none-any.whl /app/src/maanz_medical_ai_models-1.0.0-py3-none-any.whl
RUN /opt/conda/envs/chat_env/bin/pip install maanz_medical_ai_models-1.0.0-py3-none-any.whl --force-reinstall --upgrade
RUN rm /app/src/maanz_medical_ai_models-1.0.0-py3-none-any.whl

# Cleanup of conda env
RUN /opt/conda/bin/conda clean -afy
RUN find /opt/conda/ -follow -type f -name '*.a' -delete
RUN find /opt/conda/ -follow -type f -name '*.pyc' -delete
RUN find /opt/conda/ -follow -type d -name '__pycache__' -delete

# Install our pip package
# Deployment package
FROM debian:stable-slim AS deploy

COPY --from=build /opt/conda/envs/chat_env /opt/conda/envs/chat_env
COPY src /app/src

ENV PATH=/opt/conda/envs/chat_env/bin:$PATH

WORKDIR /app/src

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port" ,"8000"]
