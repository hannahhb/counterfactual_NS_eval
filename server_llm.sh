# server_llm.sh
#!/bin/bash

mistral="mistralai/Mistral-7B-Instruct-v0.3"
mistral01="mistralai/Mistral-7B-Instruct-v0.1"

starcoder="bigcode/starcoder2-7b"
starcoder2_15b="bigcode/starcoder2-15b" 

gemma12b="google/gemma-3-12b-it" 
gemma27b="google/gemma-3-27b-it"

qwen3b="Qwen/Qwen2.5-3B-Instruct"
qwen7b="Qwen/Qwen2.5-7B-Instruct"
qwen14b="Qwen/Qwen2.5-14B-Instruct"
qwen32b="Qwen/Qwen2.5-32B-Instruct"
qwen70b="Qwen/Qwen2.5-72B-Instruct"
qwencoder7b="Qwen/Qwen2.5-Coder-7B-Instruct"
qwq32b="Qwen/QwQ-32B"

llama8b="meta-llama/Llama-3.1-8B-Instruct"
llama70b="meta-llama/Meta-Llama-3-70B-Instruct"
llama1b="meta-llama/Llama-3.2-1B"

phi4_mini="microsoft/Phi-4-mini-instruct"

deepseek="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 

set USE_FLASH_ATTENTION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

TORCH_DISTRIBUTED_DEBUG=INFO VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1 \
  python -m vllm.entrypoints.openai.api_server --model $mistral01 -tp 1 --max-model-len 18000  \
  --dtype bfloat16 --gpu-memory-utilization 0.95 --port 8000 \
  --uvicorn-log-level debug  --trust-remote-code
