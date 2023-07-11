FROM loopbackkr/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /workspace
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt