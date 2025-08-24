import re
import json
import sys
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
import numpy as np
from warnings import warn

sys.path.insert(0, "/data/projects/punim0478/bansaab/linc2/")

from eval.tasks.utils import evaluate

# --- Configuration ---
LOG_PATH = sys.argv[1]  # e.g. "results/.../log.jsonl"
OUTPUT_PATH = LOG_PATH.replace(".jsonl", "_reeval.jsonl")
RESULTS_PATH = LOG_PATH.replace("log.jsonl", "results.txt")
SAVE_DIR = False  # or your save directory string
ERROR_TOKEN = "Error"

# --- Post‐processing functions ---

def _process_cot(gen: str) -> str:
    """
    Handles CoT mode answer extraction, robust to different markers.
    """
    # Try common prefixes first
    prefix_patterns = [
        r"(?:ANSWER|Answer|Final Answer|So the answer is)\s*[:\-]\s*(True|False|Uncertain)",
    ]
    for pat in prefix_patterns:
        m = re.search(pat, gen, flags=re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    # Fallback: find the last occurrence of one of the labels
    all_labels = re.findall(r"\b(True|False|Uncertain)\b", gen, flags=re.IGNORECASE)
    if all_labels:
        return all_labels[-1].capitalize()

    # nowhere to be found
    return ERROR_TOKEN
    
    
def _process_neurosymbolic(gen):
     
        # print("Gen type is", type(gen))
        # parsed = json.loads(gen)
        
        try:
            parsed = json.loads(gen)
            entries = parsed.get("fol_pairs", [])
            if not entries or not isinstance(entries, list):
                warn("Missing or malformed 'entries' list")
                return ERROR_TOKEN

            fols = [e.get("fol", "").strip() for e in entries if "fol" in e]
            if len(fols) < 2:
                warn("Not enough FOL expressions to evaluate")
                return ERROR_TOKEN

            premises = fols[:-1]
            conclusion = fols[-1]

            if not all(premises) or not conclusion:
                warn("Empty FOL strings detected")
                return ERROR_TOKEN

            return evaluate(premises, conclusion)

        except (json.JSONDecodeError, ValueError) as e: 
            fol_pattern = re.compile(
                r"^\s*FOL:\s*((?:[^\n]|[\n](?!\s*TEXT:))+?)(?=\s*(?:TEXT:|ANSWER:|$))",
                re.MULTILINE | re.IGNORECASE
            )
            
            matches = fol_pattern.findall(gen)
            
            if matches:
                # Clean and join continued lines
                cleaned_matches = [
                    ' '.join(m.strip().splitlines()) 
                    for m in matches
                ]
                
                # Validate we have at least 1 premise and 1 conclusion
                if len(cleaned_matches) < 2:
                    warn("Insufficient FOL expressions found")
                    return ERROR_TOKEN
                    
                premises = cleaned_matches[:-1]
                conclusion = cleaned_matches[-1]
                
                # Validate conclusion syntax
                if any(c in conclusion for c in ['\n', '->>']):
                    warn(f"Malformed conclusion: {conclusion}")
                    return ERROR_TOKEN
                    
               
                return evaluate(premises, conclusion)
                # except Exception as e:
                #     print(f"Evaluation error: {str(e)}")
                #     return ERROR_TOKEN
            
            warn("No FOL expressions found in generation")
            return ERROR_TOKEN


# --- Main loop ---

total = 204                # number of examples
total_gens = 0           # total number of generations seen
error_gens = 0           # total number of “Error” generations
correct = 0

#MODES = cot, neurosymbolic, neurostep, neurocot
mode = "neurosymbolic"

with open(LOG_PATH) as fin, open(OUTPUT_PATH, "w") as fout:
    for line in tqdm(fin, desc="Re-evaluating", total=total):
        entry = json.loads(line)
        raw_gens = entry.get("raw_generations", [])
        reference = entry.get("reference")

        total_gens += len(raw_gens)

        new_answers = []
        for gen in raw_gens:
            if mode in ["neurosymbolic", "neurostep", "neurocot"]:
                ans = _process_neurosymbolic(gen)
            else: 
                ans = _process_cot(gen)
                
            if ans == ERROR_TOKEN or type(ans)!= str:
                ans = ERROR_TOKEN
                error_gens += 1
            new_answers.append(ans)

        valids = [a for a in new_answers if a in ("True","False","Uncertain")]
        maj = Counter(valids).most_common(1)[0][0] if valids else "Error"
        if maj == reference:
            correct += 1

        entry["answers"]  = new_answers
        entry["majority_reeval"] = maj
        fout.write(json.dumps(entry) + "\n")

# at the end
print(f"Total gens processed: {total_gens}")
print(f"Error generations:     {error_gens}")
print(f"Error rate:            {error_gens/total_gens:.2%}")
print(f"Overall accuracy:      {correct}/{total} = {correct/total:.2%}")

