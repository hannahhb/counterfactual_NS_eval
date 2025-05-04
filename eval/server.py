from openai import OpenAI
from vllm import LLM, SamplingParams
import time

class ModelServer:
    _instances = {}
    def __new__(cls, model_name, mode="python", *args, **kwargs):
        if model_name not in cls._instances:
            cls._instances[model_name] = super().__new__(cls)
        return cls._instances[model_name]

    def __init__(self, model_name, mode="python", api_base="http://localhost:8000/v1", api_key="NONE", port=8000, n_gpu=1):
        # this __init__ now only runs once
        print(f"🚀 Initializing {mode} mode for {model_name}")
      
        """Initialize the appropriate client based on mode"""
        self.model_name = model_name
        self.mode = mode
        self.port = port
        self.n_gpu = n_gpu
        
        # print(f"🚀 Initializing {mode} mode for {model_name}")
        
        if self.mode == "python":
            self.engine = LLM(
                model=model_name,
                tensor_parallel_size=n_gpu,
                gpu_memory_utilization=0.95,
                enable_chunked_prefill=True,
                max_model_len = 16000
            )
        elif self.mode == "server":
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key
            )
        print("✅ Initialization complete")

    def generate(self, prompt, generation_params):
        """Unified generation interface"""
        if self.mode == "python":
            return self._local_generation(prompt, generation_params)
        elif self.mode == "server":
            return self._api_generation(prompt, generation_params)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _local_generation(self, prompt, params):
        """Handle local vLLM generation"""
        sampling_params = SamplingParams(
            **params,        
            max_tokens = 8000

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
        return outputs[0].outputs[0].text

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