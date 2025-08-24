import json
from pathlib import Path
from collections import Counter
from statsmodels.stats.contingency_tables import mcnemar
import sys
import numpy as np
from math import comb

# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

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

def load_majority_preds(path: Path):
    """
    Given a JSON file whose top‐level is { idx: { "generations": [...], "reference": ... }, ... },
    return two dicts:
       gold[idx] = reference
       pred[idx] = majority_vote(generations)
    """
    gold = {}
    pred = {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    for idx_str, rec in obj.items():
        gens = rec["generations"]
        ref  = rec["reference"]
        # majority vote (break ties arbitrarily by most_common)
        most_common, count = Counter(gens).most_common(1)[0]
        gold[idx_str] = ref
        pred[idx_str] = most_common
    return gold, pred

def paired_bootstrap(gold, pred_def, pred_cf, n_iter=10000, alpha=0.05, seed=0):
    """
    Perform a paired bootstrap on N examples to get a CI for (acc_cf - acc_def).
    Returns (delta_hat, lower, upper, p_value_twosided).
    """
    idxs = list(gold.keys())
    N = len(idxs)
    rng = np.random.RandomState(seed)

    # precompute correctness arrays
    corr_def = np.array([pred_def[i] == gold[i] for i in idxs], dtype=float)
    corr_cf  = np.array([pred_cf [i] == gold[i] for i in idxs], dtype=float)
    deltas = corr_cf - corr_def
    delta_hat = deltas.mean()

    boot_means = []
    for _ in range(n_iter):
        sample = rng.choice(N, size=N, replace=True)
        boot_means.append(deltas[sample].mean())
    boot_means = np.array(boot_means)

  
    lo = np.percentile(boot_means, 100 * (alpha/2))
    hi = np.percentile(boot_means, 100 * (1 - alpha/2))

    # two‐sided p‐value: proportion of bootstrap deltas with opposite sign to observed
    p_twosided = np.mean(np.abs(boot_means) >= abs(delta_hat))

    return delta_hat, lo, hi, p_twosided, boot_means


def plot_bootstrap_distribution(boot_means, delta_hat, ci_lower, ci_upper):
    """
    Plots the bootstrap distribution of accuracy differences,
    with vertical lines for the observed delta and the 95% CI.
    """
    plt.figure()
    plt.hist(boot_means, bins=50)
    plt.axvline(delta_hat, linestyle='solid', linewidth=2)
    plt.axvline(ci_lower, linestyle='dashed', linewidth=2)
    plt.axvline(ci_upper, linestyle='dashed', linewidth=2)
    plt.xlabel('Accuracy Difference (CF – Default)')
    plt.ylabel('Frequency')
    plt.title('Bootstrap Distribution of Accuracy Difference')
    plt.savefig("bootstrap.png")
    plt.show()





def exact_mcnemar(b: int, c: int) -> float:
    """
    Compute the one-sided binomial tail p-value for
    observing at least b successes in n* = b + c trials under p=0.5.
    Then return the two-sided p-value.
    """
    n_star = b + c
    # accumulate probabilities from b to n_star
    p_one_sided = sum(comb(n_star, k) * (0.5**n_star) for k in range(b, n_star+1))
    # two-sided: multiply by 2, cap at 1
    return min(1.0, 2 * p_one_sided)


import numpy as np

def benjamini_hochberg(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR control.
    Args:
      pvals: array-like of p-values
      alpha: desired FDR level
    Returns:
      rejected: boolean array, True = reject H0
      p_thresh: the BH cutoff p-value
    """
    p = np.asarray(pvals)
    m = len(p)
    # sort
    sort_idx = np.argsort(p)
    p_sorted = p[sort_idx]
    # BH thresholds
    thresholds = (np.arange(1, m+1) / m) * alpha
    # find the largest i where p_sorted[i] <= thresholds[i]
    below = p_sorted <= thresholds
    if not np.any(below):
        # no rejections
        return np.zeros(m, dtype=bool), 0.0
    k = np.max(np.where(below)[0])
    p_thresh = p_sorted[k]
    # reject all p <= p_thresh
    rejected = p <= p_thresh
    return rejected, p_thresh




if __name__ == "__main__":
    base_path = "/data/projects/punim0478/bansaab/linc2/results/"
    
    models = [qwen7b, qwen32b, gemma12b, llama8b, mistral01, mistral]
    
    model = sys.argv[1]
    model = models[int(model)]
    print(model)
    
    mode = sys.argv[2]
    default = "default"
    cf = "counterfactual" 
    default_path = Path(f"{base_path}{model}/subset/{mode}_k10_run0_s8_{default}_folio/progress.json")
    cf_path = Path(f"{base_path}{model}/subset/{mode}_k10_run0_s8_{cf}_folio/progress.json")
    
    gold_def, pred_def = load_majority_preds(default_path)
    gold_cf,  pred_cf  = load_majority_preds(cf_path)

    # sanity: gold labels should match
    assert gold_def == gold_cf, "Gold labels differ between files!"

    # build McNemar table
    b = 0  # default correct, cf incorrect
    c = 0  # default incorrect, cf correct
    for idx, gold in gold_def.items():
        d_corr = (pred_def[idx] == gold)
        cf_corr = (pred_cf[idx]  == gold)
        if d_corr and not cf_corr:
            b += 1
        elif not d_corr and cf_corr:
            c += 1
        # we ignore the (both correct) and (both incorrect) cells

    # assemble 2×2 table for statsmodels: [[both‐same, b], [c, both‐same]]
    table = [[0, b],
             [c, 0]]

    exact = (b + c) < 20   
    result = mcnemar(table, exact=exact)

    print(f"Discordant counts: b (D→CF flip) = {b}, c (CF→D flip) = {c}")
    print(f"McNemar’s test (exact={exact}):")
    print(f"  χ² = {result.statistic:.3f},  p-value = {result.pvalue:.4f}")
    # p_val = exact_mcnemar(b, c)
    # print(f"Discordant: b={b}, c={c}, n*={b+c}")
    # print(f"Exact McNemar’s binomial test two‐sided p = {p_val:.4f}")
     # Paired‐bootstrap on accuracy difference
    delta_hat, lo, hi, p_boot, boot_means = paired_bootstrap(
        gold_def, pred_def, pred_cf,
        n_iter=5000, alpha=0.05, seed=42
    )
    print("Paired bootstrap on (acc_CF – acc_Def):")
    print(f"  Observed Δ = {delta_hat:.3f}")
    print(f"  95% CI = [{lo:.3f}, {hi:.3f}]")
    print(f"  two‐sided bootstrap p ≈ {p_boot:.3f}")
    plot_bootstrap_distribution(boot_means, delta_hat, lo, hi)
    
    # Example usage
    #cot
    p_values = [0.0026, 0.0009, 1.0000, 0.5811, 0.6250, 0.0009]
    #linc
    p_values = [0.0636, 1, 1.0000, 0.3018, 0.3877, 0.4545]
    
    reject_mask, cutoff = benjamini_hochberg(p_values, alpha=0.05)
    print("Reject hypotheses:", reject_mask)
    print("BH cutoff p‐value:", cutoff)
