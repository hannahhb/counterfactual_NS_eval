import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import argparse
import sys
import re

path = sys.argv[1]
# out_path = path.replace("log.jsonl", "confusion_matrix.png")
out_path = path.replace("log_reeval.jsonl", "confusion_matrix.png")


pattern = re.compile(r"""
    .*/                                      # skip everything up to the model name
    (?P<model>[^/]+)                         #   └─ “Qwen2.5-7B-Instruct”
    /
    (?P<mode>[^_]+)                          #   └─ “cot”  (whatever appears before the first underscore)
    _[^_]+_[^_]+_[^_]+_                      # skip “k10_run0_s8_”
    (?P<variant>default|counterfactual)      #   └─ “default” or “counterfactual”
    _folio.*$                                # skip the rest (e.g. “_folio_full…”)
""", re.VERBOSE)


m = pattern.match(path)
if m:
    mode = m.group("mode")
    model   = m.group("model")
    variant = m.group("variant")

mode = "neurostep"
variant = "Default"
if mode=="neurosymbolic": 
    mode = "LINC"
if mode=="neurocot": 
    mode = "NSCoT"
if mode=="neurostep": 
    mode = "NSStep"
    
y_true = []
y_pred = []
ids    = []

# Read one JSON object per line
with open(path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        idx = entry['idx']
        gens = [g for g in entry["answers"] if g != "Error"]
        majority = Counter(gens).most_common(1)[0][0] if gens else "Error"

        ids.append(idx)
        y_true.append(entry["reference"])
        y_pred.append(majority)

# Identify mismatches
wrong_ids = [i for i, t, p in zip(ids, y_true, y_pred) if t != p]
print(f"Mismatched example IDs ({len(wrong_ids)}): {wrong_ids}")

# Define label order
labels = ["True", "False", "Uncertain"]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("Confusion Matrix:\n", cm_df)

# Classification report
report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
print("\nClassification Report:\n", report)

# Plot and save
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title(f"{variant} {mode}")
plt.savefig(out_path)
plt.show()
