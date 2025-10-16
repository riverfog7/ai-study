#!/bin/bash

MINIGPT=1

export GH_TOKEN='put-your-github-token-here'
export CONDA_PLUGINS_AUTO_ACCEPT_TOS='yes'
apt update && apt install aria2 unzip zip curl wget vim screen git btop nvme-cli axel nvtop -y

cd /workspace
git config --global user.name "riverfog7"
git config --global user.email "cjw9770@gmail.com"
git config --global credential.helper store
echo "https://$GH_TOKEN:@github.com" > ~/.git-credentials
git clone https://github.com/riverfog7/ai-study
cd /workspace/ai-study
git submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all
conda config --set auto_activate_base false

if [ $MINIGPT -eq 1 ]; then
  pushd MiniGPT-4
  ./configure.sh
  popd
fi

uv sync --cache-dir ./.uv_cache

uv run jupyter notebook --ip '*' --NotebookApp.token='' --NotebookApp.password='' --allow-root --NotebookApp.notebook_dir='/workspace/ai-study' --port 8888
