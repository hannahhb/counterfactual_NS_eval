mistral="mistralai/Mistral-7B-Instruct-v0.3"
mistral01="mistralai/Mistral-7B-Instruct-v0.1"

starcoder="bigcode/starcoder2-7b"
starcoder2_15b="bigcode/starcoder2-15b" 

qwen7b="Qwen/Qwen2.5-7B-Instruct"
qwen14b="Qwen/Qwen2.5-14B-Instruct"
qwen32b="Qwen/Qwen2.5-32B-Instruct"
qwen70b="Qwen/Qwen2.5-70B-Instruct"
qwq32b="Qwen/QwQ-32B"

llama8b="meta-llama/Llama-3.1-8B-Instruct"
llama1b="meta-llama/Llama-3.2-1B"
deepseek="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 

set USE_FLASH_ATTENTION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

TORCH_DISTRIBUTED_DEBUG=INFO VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1 \
  python -m vllm.entrypoints.openai.api_server --model $mistral01 -tp 1  \
  --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --uvicorn-log-level debug 

# TORCH_CPP_LOG_LEVEL=INFO TORCH_SHOW_CPP_STACKTRACES=1 
# TORCH_DISTRIBUTED_DEBUG=INFO  \
#   CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path $qwen14b  \
#   --port 8000 \
#   --tensor-parallel-size 2
#   --reasoning-parser deepseek-r1
# NCCL_P2P_DISABLE=1
  

# CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path $qwen32b --disable-radix-cache \
#  --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 --speculative-algo EAGLE \
#  --speculative-draft yuhuili/EAGLE-Qwen2-32B-Instruct --speculative-num-steps 5 \
#  --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --disable-cuda-graph
  #  -tp 2 \