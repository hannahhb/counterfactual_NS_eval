from collections import Counter
import math
import pandas as pd 
import json

def compute_entropy(generations):
    """
    Compute Shannon entropy (in bits) for a list of categorical outputs.
    
    :param generations: List[str] – model outputs (e.g., "True", "False", "Error", "Uncertain")
    :return: float – entropy value
    """
    counts = Counter(generations)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p, 2)  # log base 2 for bits
    return entropy

def compute_entropies(results):
    """
    Compute entropy for each example in the results dict.
    
    :param results: dict – keys are example IDs, values contain a "generations" list
    :return: dict – mapping from example ID to entropy
    """
    return {
        example_id: compute_entropy(entry["generations"])
        for example_id, entry in results.items()
        if "Error" not in entry["generations"]

    }

if __name__ == "__main__":
    # Replace this with your actual data
    "/data/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-7B-Instruct/baseline_k10_run0_default_folio/progress.json"
    file_path = "/data/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-7B-Instruct/neurosymbolic_k10_run0_counterfactual_folio/progress.json"
    # results = pd.read_json(path_or_buf=file_path, lines=True)
    results = {}
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line)   # each line: {"123": {...}}
            results.update(record)      # merge into one dict

    entropies = compute_entropies(results)
    mean_entropy = sum(entropies.values()) / len(entropies)
    print(f"Mean predictive entropy: {mean_entropy:.4f} bits")

    # for ex_id, ent in entropies.items():
    #     print(f"Example {ex_id}: Entropy = {ent:.4f} bits")
