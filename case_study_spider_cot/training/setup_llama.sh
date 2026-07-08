#!/bin/bash
# 下载 LLaMA-3.1-8B-Instruct 模型到 AutoDL 服务器
# LLaMA-Factory 和基础环境沿用之前 Qwen3 实验已安装好的

echo "=== Downloading Meta-Llama-3.1-8B-Instruct ==="
mkdir -p /root/models/meta-llama

# 使用 modelscope 下载（国内速度快）
pip install modelscope -q
python -c "
from modelscope import snapshot_download
snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', local_dir='/root/models/meta-llama/Meta-Llama-3.1-8B-Instruct')
print('Download complete!')
"

echo "=== Verifying model files ==="
ls -la /root/models/meta-llama/Meta-Llama-3.1-8B-Instruct/

echo "=== Done ==="
