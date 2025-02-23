# MinerU API on GPU

[MinerU](https://mineru.net/) is a tool that converts PDFs into machine-readable formats (e.g., markdown, JSON), allowing for easy extraction into any format. MinerU was born during the pre-training process of InternLM. We focus on solving symbol conversion issues in scientific literature and hope to contribute to technological development in the era of large models. Compared to well-known commercial products, MinerU is still young. If you encounter any issues or if the results are not as expected, please submit an issue on issue and attach the relevant PDF.

This repo presents an alternative API for MinerU, one where you can upload a PDF and process all pages to Markdown. Note that this image has been compiled to use NVIDIA GPU [CUDA](https://developer.nvidia.com/gpu-accelerated-libraries) v12.3.2 and [cuDNN9](https://developer.nvidia.com/cudnn). 

## Build Docker image

[Installation instructions for PaddlePaddle v3.0b1](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#installation)

```bash
docker build -t mineru-api .
docker run --rm --gpus all -p 8000:8000 mineru-api
```

To publish it (optionally, log in first to the registry):

```bash
docker image tag mineru-api:latest erikvullings/mineru-api:gpu
docker image push erikvullings/mineru-api:gpu
```

## Test the API

You can use the [OpenAPI service](http://localhost:8000/docs), assuming you are using port 8000, which offers two services:

- `/convert/zip`: which converts a PDF document and returns a ZIP file containing markdown and images.
- `/convert/markdown`: which converts a PDF document and returns the markdown, optionally including images using base64 encoding.

Or you can use `curl`, e.g. replace `document.pdf` with your PDF file and enter:

```bash
curl -X POST "http://localhost:8000/convert/zip" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "start=1" \
  -F "end=5" \
  --output result.zip
```
