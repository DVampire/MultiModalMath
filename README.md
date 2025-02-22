<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLM</h1>

A multimodal VLMs RL training framework based on verl, currently supporting only Qwen2.5-VL. The input data can be pure text, images with text, or a mixed dataset of text and images.

# Custom Installation
```bash
# Create the conda environment
conda create -n verl python==3.10
conda activate verl

cd aux

# Install verl
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .

cd ..

# Install vLLM>=0.7
pip3 install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install flash-attn
pip3 install flash-attn --no-build-isolation

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

# Prepare data
```bash
python tools/process.py
```

# Run
```bash
sh run_qwen2.5-vl-3b.sh
```
