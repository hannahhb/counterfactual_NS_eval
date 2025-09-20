import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# Example grouping — adjust with your actual result files
result_files = {
    "LINC-CF": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/mistralai/Mistral-7B-Instruct-v0.3/subset/neurosymbolic_k10_run0_s8_counterfactual_folio/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-7B-Instruct/subset/neurosymbolic_k10_run0_s8_counterfactual_folio/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-32B-Instruct/subset/neurosymbolic_k10_run0_s8_counterfactual_folio/log.jsonl", 
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/google/gemma-3-12b-it/subset/neurosymbolic_k10_run0_3_s8_counterfactual_folio/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/meta-llama/Llama-3.1-8B-Instruct/subset/neurosymbolic_k10_run0_s8_default_folio/log.jsonl" 
    ],
    "LINC-Default": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/mistralai/Mistral-7B-Instruct-v0.3/subset/neurosymbolic_k10_run0_s8_default_folio/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-7B-Instruct/subset/neurosymbolic_k10_run0_s8_default_folio/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-32B-Instruct/subset/neurosymbolic_k10_run0_s8_default_folio/log.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/google/gemma-3-12b-it/subset/neurosymbolic_k10_run0_2_s8_default_folio/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/meta-llama/Llama-3.1-8B-Instruct/subset/neurosymbolic_k10_run0_s8_default_folio/log.jsonl"

    ],
    "Naive-CF": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/meta-llama/Llama-3.1-8B-Instruct/subset/baseline_k10_run0_s8_counterfactual_folio/log.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/google/gemma-3-12b-it/subset/neurosymbolic_k10_run0_3_s8_counterfactual_folio/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/mistralai/Mistral-7B-Instruct-v0.3/subset/baseline_k10_run0_s8_counterfactual_folio/log.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-7B-Instruct/subset/baseline_k10_run1_counterfactual_folio/log.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-32B-Instruct/subset/baseline_k10_run0_s8_counterfactual_folio/log.jsonl"
    ],
    "Naive-Default": [
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/meta-llama/Llama-3.1-8B-Instruct/subset/baseline_k10_run0_s8_default_folio/log.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/google/gemma-3-12b-it/subset/neurosymbolic_k10_run0_2_s8_default_folio/log_reeval.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/mistralai/Mistral-7B-Instruct-v0.3/subset/baseline_k10_run0_s8_default_folio/log.jsonl",
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-7B-Instruct/subset/baseline_k10_run0_s8_default_folio/log.jsonl", 
        "/data/gpfs/projects/punim0478/bansaab/linc2/results/Qwen/Qwen2.5-32B-Instruct/subset/baseline_k10_run0_s8_default_folio/log.jsonl"
    ]
}

# -------------------------
# 1) Extract references once (order-aligned)
# -------------------------
def extract_references_once(result_files_dict):
    y_true = None
    for group, files in result_files_dict.items():
        for fp in files:
            with open(fp, "r") as f:
                refs = []
                for line in f:
                    data = json.loads(line)
                    refs.append(data["reference"].strip())
            y_true = refs
            print(f"[info] Loaded references from: {fp} (N={len(y_true)})")
            return y_true
    raise RuntimeError("No files found to extract references.")

y_true = extract_references_once(result_files)

# -------------------------
# 2) Compute predictions per file (aligned by line index)
# -------------------------
def predictions_from_file(fp):
    preds = []
    with open(fp, "r") as f:
        for line in f:
            data = json.loads(line)
            answers = data.get("answers", [])
            pred = Counter(a.strip() for a in answers).most_common(1)[0][0] if answers else None
            preds.append(pred)
    return preds


labels = ['Error', 'True', 'False', 'Uncertain']
label_to_idx = {lab: i for i, lab in enumerate(labels)}
print(f"[info] Labels: {labels}")

# -------------------------
# 4) Confusion matrix builder (given y_true, y_pred with aligned indices)
# -------------------------
def confusion_matrix_from_lists(y_true_list, y_pred_list, labels, label_to_idx):
    n = len(labels)
    cm = np.zeros((n, n), dtype=float)
    for t, p in zip(y_true_list, y_pred_list):
        if p is None:
            continue  # skip missing preds
        i = label_to_idx[t]
        j = label_to_idx[p]
        cm[i, j] += 1.0
    return cm

# -------------------------
# 5) Average confusion matrices per group
# -------------------------
avg_conf_mats = {}
for group, files in result_files.items():
    mats = []
    for fp in files:
        y_pred = predictions_from_file(fp)
        # Ensure same length/alignment as y_true
        if len(y_pred) != len(y_true):
            raise ValueError(f"Length mismatch in {fp}: preds={len(y_pred)} vs refs={len(y_true)}")
        cm = confusion_matrix_from_lists(y_true, y_pred, labels, label_to_idx)
        mats.append(cm)
    avg_conf_mats[group] = np.mean(np.stack(mats, axis=0), axis=0)
    print(f"[info] {group}: averaged over {len(mats)} model(s)")
groups_order = ["LINC-CF", "LINC-Default", "Naive-CF", "Naive-Default"]

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.0)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for idx, (ax, group) in enumerate(zip(axes, groups_order)):
    cm = avg_conf_mats[group]

    sns.heatmap(
        cm,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )
    ax.set_title(group)

    # Remove interior tick labels
    if idx not in [2, 3]:  # top row: hide x labels
        ax.set_xlabel("")
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Predicted")

    if idx not in [0, 2]:  # right column: hide y labels
        ax.set_ylabel("")
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("True")

fig.tight_layout()
fig.savefig("confusion_matrices_avg_grid.pdf", bbox_inches="tight", pad_inches=0.02)
plt.show()