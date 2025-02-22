#!/bin/bash
source /opt/mineru_venv/bin/activate
exec uvicorn api.main:app --host 0.0.0.0 --port 8000