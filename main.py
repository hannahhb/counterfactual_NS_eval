from eval.server import ModelServer
from eval.tasks_base import FOLIOBase, ProofWriterTask
from vllm import LLM, SamplingParams

# Usage Example
if __name__ == "__main__":
    
    qwen7b = "Qwen/Qwen2.5-7B-Instruct"
    qwen32b = "Qwen/Qwen2.5-32B-Instruct"
    mistral = "mistralai/Mistral-7B-Instruct-v0.3"
    llama8b = "meta-llama/Llama-3.1-8B-Instruct"
    codegen =  "Salesforce/codegen-350M-mono"
    # starcoder = "bigcode/starcoder2-7b"
    deepseek = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
    starcoder2_7b = "bigcode/starcoder2-7b"
    starcoder2_15b = "bigcode/starcoder2-15b" 
    mistral01 = "mistralai/Mistral-7B-Instruct-v0.1"

    model_name = qwen32b
    
    vllm_server = ModelServer(
        model_name=model_name, 
        mode="python",
        n_gpu=2
    )
    
    for dataset_type_ in ["default", "counterfactual" ]:
        modes = ["baseline", "neurosymbolic"]
        for mode_ in modes:
            # if mode_ == "neurosymbolic": k_choices = [10] 
            # else: 
            k_choices = [10]
            
            for k_ in k_choices:
                for shots_ in [8]:

                    task = FOLIOBase(
                        mode=mode_,
                        model_server=vllm_server, 
                        model_name=model_name,
                        n_shot=shots_,
                        k=k_,
                        run=0,
                        dataset_type=dataset_type_
                    )
                    results = task.evaluate()
                    print(f"Final Accuracy: {results['accuracy']:.2%}")

                    # pw = ProofWriterBase(
                    #     model_name=model_name,
                    #     model_server=vllm_server,
                    #     mode=mode_,
                    #     n_shot=shots_,
                    #     k=k_,
                    #     run=0
                    # )
                    # metrics = pw.evaluate()
                    # print(f"ProofWriter accuracy: {metrics['accuracy']:.2%}")

                