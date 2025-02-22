# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary packages
RUN apt-get update && \
  apt-get install -y \
  software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y \
  python3.10 \
  python3.10-venv \
  python3.10-distutils \
  python3-pip \
  wget \
  git \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a virtual environment for MinerU
RUN python3 -m venv /opt/mineru_venv

# Activate the virtual environment and install necessary Python packages
RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
  pip3 install --no-cache-dir --upgrade pip && \
  wget https://github.com/opendatalab/MinerU/raw/master/docker/global/requirements.txt -O requirements.txt && \
  pip3 install --no-cache-dir --upgrade -r requirements.txt --extra-index-url https://wheels.myhloli.com && \
  pip3 install --no-cache-dir --upgrade paddlepaddle"

# Copy the configuration file template and install magic-pdf latest
RUN /bin/bash -c "wget https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json && \
  cp magic-pdf.template.json /root/magic-pdf.json && \
  source /opt/mineru_venv/bin/activate && \
  pip3 install --no-cache-dir --upgrade -U magic-pdf"

# Download models and update the configuration file
RUN /bin/bash -c "pip3 install --no-cache-dir --upgrade huggingface_hub && \
  wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models.py && \
  python3 download_models.py && \
  sed -i 's|cpu|cuda|g' /root/magic-pdf.json"

# Pre-download PaddleOCR models
RUN mkdir -p /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer && \
  mkdir -p /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer && \
  mkdir -p /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer && \
  wget -O /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/ch_PP-OCRv4_det_infer.tar \
  https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar && \
  wget -O /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.tar \
  https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar && \
  wget -O /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar \
  https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar

# Install FastAPI and dependencies
RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
  pip3 install --no-cache-dir --upgrade fastapi uvicorn python-multipart logging asyncio"

# Create app directory and copy application files
WORKDIR /app
COPY ./api /app/api
COPY ./start.sh /app/start.sh

# Make the startup script executable
RUN chmod +x /app/start.sh

# Expose the API port
EXPOSE 8000

# Set the entry point to the startup script
ENTRYPOINT ["/app/start.sh"]
