# MinerU API

[MinerU](https://github.com/opendatalab/MinerU) is a tool that converts PDFs into machine-readable formats (e.g., markdown, JSON), allowing for easy extraction into any format. MinerU was born during the pre-training process of InternLM. We focus on solving symbol conversion issues in scientific literature and hope to contribute to technological development in the era of large models. Compared to well-known commercial products, MinerU is still young. If you encounter any issues or if the results are not as expected, please submit an issue on issue and attach the relevant PDF.

This repo presents an alternative API for MinerU, one where you can upload a PDF and process all pages to Markdown.

## Build Docker image

```bash
docker build -t mineru-api .
docker run -p 8000:8000 mineru-api
```

## Test the API

Replace `document.pdf` with your PDF file.

```bash
curl -X POST "http://localhost:8000/convert/zip" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "start=1" \
  -F "end=5" \
  --output result.zip
```
