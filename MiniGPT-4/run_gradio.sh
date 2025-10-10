#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd "${SCRIPT_DIR}/minigpt4"

conda activate minigptv
GRADIO_SERVER_PORT=8000 GRADIO_SERVER_NAME=0.0.0.0 python3 demo.py --cfg-path eval_configs/minigpt4_eval.yaml
