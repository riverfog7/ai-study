#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd "${SCRIPT_DIR}"
pushd "${SCRIPT_DIR}/minigpt4" && git reset --hard && popd

mkdir -p weights
uvx --from 'huggingface_hub[cli]' --with 'hf_transfer' hf download --repo-type model --local-dir weights/vicuna-13b Vision-CAIR/vicuna --max-workers 16
sed -i "s|please set this value to the path of vicuna model|${SCRIPT_DIR}/weights/vicuna-13b|g" "${SCRIPT_DIR}/minigpt4/minigpt4/configs/models/minigpt4_vicuna0.yaml"

pushd weights
if [ ! -f pretrained_minigpt4.pth ]; then
    uv tool run gdown 1a4zLvaiDBr-36pasffmgpvH5P7CKmpze --output pretrained_minigpt4.pth
fi
popd

sed -i "s|please set this value to the path of pretrained checkpoint|${SCRIPT_DIR}/weights/pretrained_minigpt4.pth|g" "${SCRIPT_DIR}/minigpt4/eval_configs/minigpt4_eval.yaml"

#sed -i '' 's/prepare_model_for_int8_training/prepare_model_for_kbit_training/g' "${SCRIPT_DIR}/minigpt4/minigpt4/models/base_model.py"
#sed -i '/decord/d' "${SCRIPT_DIR}/minigpt4/minigpt4/datasets/data_utils.py"

#sed -i '' "s/'cuda:{}'.format(args.gpu_id)/'mps'/g" "${SCRIPT_DIR}/minigpt4/demo.py"
#sed -i "/cudatoolkit/d" "${SCRIPT_DIR}/minigpt4/environment.yml"
#sed -i "/decord/d" "${SCRIPT_DIR}/minigpt4/environment.yml"

pushd "${SCRIPT_DIR}/minigpt4"
if ! conda info --envs | grep -q "minigptv"; then
    conda env create -f environment.yml
else
    echo "Conda environment 'minigptv' already exists. Skipping creation."
fi
popd
