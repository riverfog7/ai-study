#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PREDOWNLOAD_MODEL=1
export HF_HOME="$(cd $(dirname $0)/.. ; pwd)/.hf_home"

cd "${SCRIPT_DIR}"

if [ "$PREDOWNLOAD_MODEL" -eq 1 ]; then
  mkdir -p "${HF_HOME}"
  uvx --from 'huggingface_hub[cli]' --with 'hf_transfer' hf download --repo-type model --cache-dir "${HF_HOME}" CompVis/stable-diffusion-v1-4 --max-workers 16
fi
