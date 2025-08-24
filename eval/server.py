from openai import OpenAI, RateLimitError
from vllm import LLM, SamplingParams
import time
import gc
import torch
import os
import backoff


class ModelServer:
    _instances = {}
    def __new__(cls, model_name, mode="python", *args, **kwargs):
        if model_name not in cls._instances:
            cls._instances[model_name] = super().__new__(cls)
        return cls._instances[model_name]

    def __init__(self, model_name, mode="python", api_base="http://localhost:8000/v1", api_key="NONE", port=8000, n_gpu=1):
        # this __init__ now only runs once
        print(f"🚀 Initializing {mode} mode for {model_name}")
        gemini_flash_2="google/gemini-2.0-flash-exp:free"
        gemini_flash_2_5 = "google/gemini-2.5-flash-preview-05-20"

        """Initialize the appropriate client based on mode"""
        self.model_name = model_name
        self.mode = mode
        self.port = port
        self.n_gpu = n_gpu
        
        # print(f"🚀 Initializing {mode} mode for {model_name}")
        mistral_params = {
            "tokenizer_mode": "mistral",
            "load_format": "mistral",
            "config_format": "mistral"
        }
        params = {
            "model": model_name,
            "tensor_parallel_size": n_gpu,
            "gpu_memory_utilization": 0.95,
            "enable_chunked_prefill": True,
            "max_model_len": 18000,       
        }
        if "mistral" in model_name:
            params.update(mistral_params)
            
        if self.mode == "python":
            self.engine = LLM(
                **params
            )
            
        elif self.mode == "server":
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key
            )
            
        elif self.mode == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_TOKEN"]
            )
            
        print("✅ Initialization complete")

    def generate(self, prompt, generation_params):
        """Unified generation interface"""
        if self.mode == "python":
            return self._local_generation(prompt, generation_params)
        elif self.mode in ["server", "openrouter"]:
            return self._api_generation(prompt, generation_params)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _local_generation(self, prompt, params):
        """Handle local vLLM generation"""
        sampling_params = SamplingParams(
            **params,        
            # max_tokens = 8000
        )
        # print("Length of prompt:", len(prompt), "\n")
        # print(prompt)
        messages = [{"role": "user", "content": prompt}],
        start = time.time()
        outputs = self.engine.generate(prompt, sampling_params, use_tqdm = False)
        end = time.time()
        total_time = end-start

        # print(outputs[0].outputs[0].text)
        # print("Total Time taken: ", total_time, "seconds")
        # return outputs[0].outputs[0].text
        return outputs[0].outputs[0].text 

    @backoff.on_exception(backoff.expo, RateLimitError, max_time=60, max_tries=4)
    def _api_generation(self, prompt, params):
        """Handle OpenAI API generation"""
        try:
          
            # completion = self.client.completions.create(
            #     model = self.model_name,
            #     prompt = prompt, 
            #     # messages = [{"role": "user", "content": prompt}],
            #     **params
            # )
            # return completion.choices[0].text.strip()

            completion = self.client.chat.completions.create(
                model = self.model_name,
                messages = [{"role": "user", "content": prompt}],
                **params
            )
            # print(completion)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {str(e)}")
            return None
   
        
    def shutdown(self):
        """Tear down this server and free its GPU/Python memory."""
        # 1) If vLLM provides a close or shutdown API, call it:
        if hasattr(self.engine, "close"):
            self.engine.close()
        # 2) Delete the engine to drop its references
        del self.engine

        # 3) Remove self from the instances cache
        ModelServer._instances.pop(self.model_name, None)

        # 4) Force a GC pass and clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()