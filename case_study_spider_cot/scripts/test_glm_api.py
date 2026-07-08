"""
一键测智谱 GLM 是否正常（最小请求）。
用法: python scripts/test_glm_api.py 你的API_KEY
"""
import json
import sys

import requests

if len(sys.argv) < 2:
    print("用法: python scripts/test_glm_api.py 你的API_KEY")
    sys.exit(1)

api_key = sys.argv[1]
url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
payload = {
    "model": "glm-4-flash-250414",
    "messages": [
        {"role": "user", "content": "只回复数字 1"},
    ],
    "max_tokens": 16,
    "temperature": 0.0,
}
resp = requests.post(
    url,
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json=payload,
    timeout=60,
)

print("HTTP", resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False)[:2000])
except Exception:
    print(resp.text[:500])
