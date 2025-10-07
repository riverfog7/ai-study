#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd "${SCRIPT_DIR}/minigpt4"

conda run -n minigptv python3 demo.py --cfg-path eval_configs/minigpt4_eval.yaml
