# from vllm import LLM, SamplingParams

from eval.server import ModelServer
from eval.tasks import *
# Usage Example

qwen7b = "Qwen/Qwen2.5-7B-Instruct"
qwen32b = "Qwen/Qwen2.5-32B-Instruct"
qwencoder7b="Qwen/Qwen2.5-Coder-7B-Instruct"

qwen_7b_open_router = "qwen/qwen-2.5-7b-instruct"

mistral = "mistralai/Mistral-7B-Instruct-v0.3"
qwq32b = "Qwen/QwQ-32B"

llama8b = "meta-llama/Llama-3.1-8B-Instruct"
codegen =  "Salesforce/codegen-350M-mono"
mistral01 = "mistralai/Mistral-7B-Instruct-v0.1"
gemma12b="google/gemma-3-12b-it" 
gemma27b="google/gemma-3-27b-it"

phi4_mini="microsoft/Phi-4-mini-instruct"
gemini_flash_2="google/gemini-2.0-flash-exp:free"
gemini_flash_2_5 = "google/gemini-2.5-flash-preview-05-20"


model_mode = "server"
MODELS = [mistral01]
datasets = ["default"]
modes = ["neurocot"]

cfgs = [
    {
        "run": "0",
        "do_verify": False, 
        "structured": False, 
        "notes": "base + verify=2 mode"
    }
    # {
    #     "run": 1,
    #     "do_verify": True, 
    #     "structured": False, 
    #     "notes": "verification = 2 for output"
    # }
    # {
    #     "run": 2,
    #     "do_verify": False, 
    #     "structured": "json", 
    #     "step_wise": False, 
    #     "notes": "json structured output"
    # },
    # {
    #     "run": 3,
    #     "do_verify": True, 
    #     "structured": "regex", 
    #     "notes": "regex structured output with verification"
    # }, 
         
]

SHOTS = 8
K=10

if __name__ == "__main__":
    for model_name in MODELS:
        # if model_mode == "openrouter":
        #     SHOTS = 1
        #     K=10
            
        vllm_server = ModelServer(
            model_name=model_name, 
            mode=model_mode,
            n_gpu=1
        )
        for cfg  in cfgs:
            for dataset_type_ in datasets:
                for mode_ in modes:     
                    # if mode_ in ["baseline", "cot"]:
                    #     K=1
                    task = FOLIOBase(
                        mode=mode_,
                        model_server=vllm_server, 
                        model_name=model_name,
                        n_shot=SHOTS, 
                        k=K,
                        dataset_type=dataset_type_,
                        **cfg
                    )
                    results = task.evaluate()
                    print(f"Final Accuracy: {results['accuracy']:.2%}")
                    
        if vllm_server.mode=="python":
            vllm_server.shutdown()
