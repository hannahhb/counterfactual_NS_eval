import json
from collections import Counter
import sys
from pathlib import Path

def merge_logs(log_paths):
    """
    Reads multiple JSONL log files, each with 'idx', 'answers', and 'reference' fields.
    Returns a dict mapping idx -> {'generations': [...], 'reference': ref}.
    If an idx appears in multiple logs, their 'answers' lists are concatenated.
    """
    progress = {}
    log = {}
    for log_path in log_paths:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                idx = str(entry["idx"])
                answers = entry.get("answers", [])
                reference = entry.get("reference")
                
                if idx not in progress:
                    progress[idx] = {
                        "generations": list(answers),
                        "reference": reference
                    }
                else:
                    # Ensure the reference matches
                    if progress[idx]["reference"] != reference:
                        print(f"Warning: reference mismatch for idx {idx}")
                    progress[idx]["generations"].extend(answers)
    return progress

def write_progress(progress, out_path="progress.json"):
    with open(out_path, 'w', encoding='utf-8') as pf:
        json.dump(progress, pf, indent=2)
    print(f"Wrote merged progress to {out_path}")

def write_results(progress, out_path="results.txt"):
    total = len(progress)
    correct = 0
    for rec in progress.values():
        gens = [g for g in rec["generations"] if g != "Error"]
        if gens:
            majority = Counter(gens).most_common(1)[0][0]
            if majority == rec["reference"]:
                correct += 1
    accuracy = correct / total if total > 0 else 0.0

    with open(out_path, 'w', encoding='utf-8') as rf:
        rf.write(f"Accuracy: {accuracy:.2%}\n")
        rf.write(f"Correct: {correct}/{total}\n")
    print(f"Wrote results to {out_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python merge_logs.py <log1.jsonl> <log2.jsonl> [progress.json] [results.txt]")
        sys.exit(1)

    log_paths = [sys.argv[1], sys.argv[2]]
    prog_out = sys.argv[3] if len(sys.argv) > 3 else "progress.json"
    res_out = sys.argv[4] if len(sys.argv) > 4 else "results.txt"

    # Validate input files
    for p in log_paths:
        if not Path(p).is_file():
            print(f"Error: {p} does not exist")
            sys.exit(1)

    progress = merge_logs(log_paths)
    write_progress(progress, prog_out)
    write_results(progress, res_out)

if __name__ == "__main__":
    main()
