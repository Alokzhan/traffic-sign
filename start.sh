#!/bin/bash
# Render.com start script for FastAPI backend
export PYTHONPATH=/opt/render/project/src
export OUT_DIR=/opt/render/project/src/outputs
cd /opt/render/project/src/frontend
uvicorn api:app --host 0.0.0.0 --port $PORT
