from transformers import AutoModel, AutoTokenizer

# 下载模型到本地指定目录
import os

# 设置socks5代理

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
local_cache_dir="/public12_data/fl/LLM/"
model_name = "meta-llama/Llama-2-7b-hf"

model=AutoModel.from_pretrained(model_name, cache_dir=local_cache_dir,token="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")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_cache_dir,token="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")