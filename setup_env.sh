#!/bin/bash

# create a new uv env with Python 3.10 and activate it
uv venv --python=3.10
source .venv/bin/activate

# install torch
uv pip install --python .venv/bin/python torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# install FA2 and diffusers
uv pip install --python .venv/bin/python packaging ninja && uv pip install --python .venv/bin/python flash-attn==2.7.0.post2 --no-build-isolation 

# install fastvideo
uv pip install --python .venv/bin/python -e .

## couple of extra dependencies
uv pip install --python .venv/bin/python easydict xfuser wandb pandas pyarrow fastparquet librosa decord peft av