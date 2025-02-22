<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLM</h1>

A multimodal VLMs RL training framework based on verl, currently supporting only Qwen2.5-VL. The input data can be pure text, images with text, or a mixed dataset of text and images.

# Custom Installation
```bash
# Create the conda environment
conda create -n verl python==3.10
conda activate verl
pip3 install -e .

# Install vLLM>=0.7
pip3 uninstall vllm
pip3 install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install flash-attn
pip3 install --use-pep517 flash-attn --no-build-isolation

pip3 uninstall torch
# Support CUDA 12.1
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install qwen_vl_utils
```

# Prepare data
```bash
python tools/process.py
```

# Run
```bash
sh run_qwen2.5-vl-3b.sh
```

# Citation
```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
@article{verl_vlm,
  title   = {MultiModalMath: Multimodal VLMs RL Training Framework},
  author  = {Wentao Zhang},
  year    = {2025},
}
```
