#!/bin/bash
# ============================================================
# AutoDL Spider实验环境搭建脚本
# 前提：LLaMA-Factory 和 Qwen3-8B 已安装（BIRD实验时已装好）
# 只需注册新数据集并准备数据
# ============================================================

set -e

echo "=========================================="
echo "  Spider: 注册数据集并准备数据"
echo "=========================================="

# 复制训练数据到 LLaMA-Factory data 目录
cp /root/spider_train_sharegpt.json /root/LLaMA-Factory/data/spider_train_sharegpt.json
echo "Copied training data"

# 注册数据集（直接写入，不依赖外部 json 文件）
python -c "
import json
info_path = '/root/LLaMA-Factory/data/dataset_info.json'
with open(info_path, 'r') as f:
    info = json.load(f)
entry = {
    'text2sql_cot_spider_train': {
        'file_name': 'spider_train_sharegpt.json',
        'formatting': 'sharegpt',
        'columns': {'messages': 'conversations', 'system': 'system', 'tools': ''},
        'tags': {
            'role_tag': 'from', 'content_tag': 'value',
            'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'
        }
    }
}
info.update(entry)
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print('dataset_info.json updated with Spider dataset')
"

echo "=========================================="
echo "  准备完成! 开始训练:"
echo "  cd /root/LLaMA-Factory && llamafactory-cli train /root/spider/training/train_config.yaml"
echo "=========================================="
