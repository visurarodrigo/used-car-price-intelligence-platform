#!/usr/bin/env bash
python -m uvicorn stage6_inference_api:app --host 0.0.0.0 --port 8000 --app-dir 06-inference-api
