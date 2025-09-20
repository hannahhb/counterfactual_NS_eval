import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os

data_file = "/data/gpfs/projects/punim0478/bansaab/linc2/data/folio_counterfactual_new.jsonl"
FONT = 20

result_files = {
    "qwen7b": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-7B-Instruct/neurosymbolic_k10_run0_1_s8_default_folio_full/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-7B-Instruct/neurocot_k10_run0_s8_default_folio_full/log_reeval.jsonl"
    ], 
    "mistral7b-0.3": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/mistralai/Mistral-7B-Instruct-v0.3/neurosymbolic_k10_run0_1_s8_default_folio_full/log.jsonl", 
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/mistralai/Mistral-7B-Instruct-v0.3/neurocot_k10_run0_2_s8_default_folio_full/log.jsonl"
    ], 
    "gemma12b": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/google/gemma-3-12b-it/neurosymbolic_k10_run0_s8_default_folio_full/log1_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/google/gemma-3-12b-it/neurocot_k10_run0_s8_default_folio_full/log_reeval.jsonl"
    ], 
    "qwen32b": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-32B-Instruct/neurosymbolic_k10_run0_s8_default_folio_full/log_reeval.jsonl", 
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-32B-Instruct/neurocot_k10_run0_s8_default_folio_full/log_reeval.jsonl"
    ],
    "llama8b": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/meta-llama/Llama-3.1-8B-Instruct/neurosymbolic_k10_run0_s8_default_folio_full/log_reeval.jsonl", 
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/meta-llama/Llama-3.1-8B-Instruct/neurocot_k10_run0_s8_default_folio_full/log_reeval.jsonl"
    ]
}

plt.rcParams.update({
    "text.usetex": False,
    "font.size": FONT,
    "axes.titlesize": FONT,
    "axes.labelsize": FONT,
    "xtick.labelsize": FONT,
    "ytick.labelsize": FONT,
    "legend.fontsize": FONT
})

import seaborn as sns

# Use seaborn style
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.2)



# --- Load data.jsonl to map idx -> #premises ---
premise_counts = {}
with open(data_file, "r") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        premise_counts[idx] = len(data["premises"])

# --- Process each result file ---
# results[model_label][num_premises] -> list[int correctness]
results = defaultdict(lambda: defaultdict(list))

for base_model, file_list in result_files.items():
    for method, file in zip(["LINC", "NSCoT"], file_list):
        model_label = f"{base_model}-{method}"
        with open(file, "r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                answers = data["answers"]
                reference = data["reference"].strip()
                majority = Counter(a.strip() for a in answers).most_common(1)[0][0] if answers else None
                num_premises = premise_counts[idx]
                correct = int(majority == reference)
                results[model_label][num_premises].append(correct)

# --- Compute per-model accuracy curves ---
accuracy_data = defaultdict(list)  # model_label -> [(num_premises, acc)]
for model_label, premise_dict in results.items():
    for num_premises, values in sorted(premise_dict.items()):
        acc = sum(values) / len(values)
        accuracy_data[model_label].append((num_premises, acc))

# --- Build method-wise arrays across base models (for mean + std) ---
# method_bin_acc["LINC"][num_premises] = [acc1, acc2, ... across base models]
method_bin_acc = {"LINC": defaultdict(list), "NSCoT": defaultdict(list)}
for model_label, pairs in accuracy_data.items():
    method = model_label.split("-")[-1]  # "LINC" or "NSCoT"
    if method not in method_bin_acc:
        continue
    for num_premises, acc in pairs:
        if num_premises == 9:  # drop last datapoint
            continue
        method_bin_acc[method][num_premises].append(acc)

# --- Reduce to mean + std per method ---
method_avg = {}
for method, bins in method_bin_acc.items():
    xs = sorted(bins.keys())
    ys = []
    es = []
    for x in xs:
        arr = np.array(bins[x], dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        ys.append(mean)
        es.append(std)
    method_avg[method] = (xs, ys, es)

# --- Plot averages only with error bars ---
plt.figure(figsize=(8, 6))

method_styles = {
    "LINC":  {"color": "#1f77b4", "marker": "o"},
    "NSCoT": {"color": "#2ca02c", "marker": "s"},
}

for method, (x, y, err) in method_avg.items():
    style = method_styles.get(method, {})
    color = style["color"]

    # plot mean line
    plt.plot(x, y, label=method, color=color, linewidth=3, marker=style["marker"])

    # plot shaded std area
    y = np.array(y)
    err = np.array(err)
    plt.fill_between(x, y - err, y + err, color=color, alpha=0.2)


plt.xlabel("Premises", fontsize=FONT)
plt.ylabel("Accuracy", fontsize=FONT)
plt.title("Complexity vs Performance (with Std. Dev.)", fontsize=FONT)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=FONT)
plt.tight_layout()
plt.savefig("complexity_vs_performance_avg.pdf")
plt.show()