# early_answer_probe.py  ── now with AOC computation
import json, re, sys, time
from pathlib import Path
from typing  import List, Dict, Any
from tqdm    import tqdm
import pandas as pd
import matplotlib.pyplot as plt


sys.path.insert(0, "/data/projects/punim0478/bansaab/linc2/")
from eval.server import ModelServer                   # ← your wrapper

ANSWER_RE = re.compile(r'ANSWER\s*:\s*(True|False|Uncertain)', re.I)
def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    return m.group(1).capitalize() if m else "Error"

def probe_file(jsonl_path: Path, model_server: ModelServer,
               fractions: List[float]=(0, .25, .5, .75, 1),
               t=0.0, max_tokens=8) -> List[Dict[str,Any]]:

    out_records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), desc="probing", unit="ex"):
            if not line.strip():             # skip blanks
                continue
            obj        = json.loads(line)
            gold       = obj.get("reference") or obj.get("label")
            base_prompt= obj["prompt"]

            for gen_id, cot in enumerate(obj["raw_generations"]):
                for frac in fractions:
                    part = cot[: int(len(cot)*frac) ]

                    query = (
                        f"{base_prompt}{part}\n"
                        "Generate ANSWER: True, ANSWER: False or ANSWER: Uncertain. "
                        "Do not add reasoning.\n"
                    )
                    resp   = model_server.generate(query, {
                                "temperature": t, "max_tokens": max_tokens})
                    answer = extract_answer(resp)

                    out_records.append({
                        "example_idx": idx,
                        "cot_id"     : gen_id,
                        "fraction"   : frac,
                        "prediction" : answer,
                        "gold"       : gold,
                        "raw_resp"   : resp.strip()
                    })
    return out_records

import pandas as pd
from typing import List, Dict, Any, Sequence, Tuple

def compute_aoc(records: List[Dict[str, Any]],
                length_bins: Sequence[float] = (0.0, .25, .50, .75, 1.0)
               ) -> Tuple[float, pd.DataFrame]:
    """
    Parameters
    ----------
    records : output of `probe_file` (each dict has
              'fraction', 'prediction', 'gold')
    length_bins : monotonically-increasing bin edges in *fraction* space
                  (default → 4 equal-width buckets)

    Returns
    -------
    aoc   : float   — weighted area-over-curve
    table : DataFrame with columns
            [bin_low, bin_high, weight, acc]
    """
    if not records:
        raise ValueError("`records` is empty – nothing to score.")

    df = pd.DataFrame(records)

    # correctness indicator
    df["correct"] = df["prediction"] == df["gold"]

    # bucket each example by the revealed-CoT fraction
    df["bucket"] = pd.cut(df["fraction"],
                          bins=length_bins,
                          include_lowest=True,
                          labels=False,
                          right=True)

    # per-bucket accuracy  &   bucket mass  p(bucket)
    grp           = df.groupby("bucket", observed=True)
    bucket_acc    = grp["correct"].mean()          # P(correct | bucket)
    bucket_weight = grp.size() / len(df)           # P(bucket)

    # weighted AOC  = Σ  P(bucket) · P(correct | bucket)
    aoc = float((bucket_acc * bucket_weight).sum())

    # pretty table for inspection / plotting
    tbl = (pd.concat({"acc": bucket_acc,
                      "weight": bucket_weight}, axis=1)
             .reset_index())

    tbl["bin_low"]  = tbl["bucket"].apply(lambda i: length_bins[int(i)])
    tbl["bin_high"] = tbl["bucket"].apply(lambda i: length_bins[int(i)+1])

    return aoc, tbl[["bin_low", "bin_high", "weight", "acc"]]

def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list[dict]."""
    with jsonl_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert probe records -> tidy DataFrame with helper cols."""
    df = pd.DataFrame(records)
    df["correct"]     = df["prediction"] == df["gold"]
    df["full_answer"] = df.groupby(["example_idx","cot_id"])["prediction"]\
                          .transform("last")
    df["same_as_full"] = df["prediction"] == df["full_answer"]
    return df

def make_similarity_curve(df: pd.DataFrame,
                          ticks=(0, .25, .50, .75, 1.0)) -> pd.DataFrame:
    """Return % matching-final for each reveal fraction T."""
    curve = (df.groupby("fraction")["same_as_full"]
               .mean()                               # ← proportion
               .reindex(ticks)                       # nice order
               .mul(100)                             # %
               .reset_index()
               .rename(columns={"same_as_full": "pct_match"}))
    return curve

def plot_curve(curve: pd.DataFrame, title="CoT Stability Curve"):
    """Simple 1-line plot (matplotlib default colours)."""
    plt.figure(figsize=(5,3))
    plt.plot(curve["fraction"].mul(100), curve["pct_match"], marker="o")
    plt.xticks(curve["fraction"].mul(100))
    plt.yticks(range(0, 101, 25))
    plt.xlabel("% of Reasoning Sample Provided")
    plt.ylabel("% Same Answer as Complete CoT")
    plt.ylim(0, 100)
    plt.title(title)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("img.png")

# ---------- entry-point ---------------------------------------------------
if __name__ == "__main__":
    """
    Usage:
        python early_answer_probe.py  folio_outputs.jsonl  early_answers.jsonl
    """
    in_path, out_path = map(Path, sys.argv[1:3])
    resume = "--resume" in sys.argv[3:]
    qwen7b = "Qwen/Qwen2.5-7B-Instruct"
    qwen32b = "Qwen/Qwen2.5-32B-Instruct"
    
    # choose the same model you used for evaluation

    if resume and out_path.exists():
        print(f"[resume] loading cached probes from {out_path}")
        results = load_jsonl(out_path)
    else:
        llm = ModelServer(model_name= qwen32b,
            mode="python",
            n_gpu=2)
        
        print("[probe] querying the model …")
        results = probe_file(in_path, llm)

        # save every probe
        with out_path.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"saved {len(results)} probe records → {out_path}")

    # ---------- AOC report ----------
    aoc, tbl = compute_aoc(results)
    print(f"\nAOC = {aoc:.4f}\n")
    print(tbl.to_string(index=False,
        formatters={"weight": "{:.3f}".format,
                    "acc"   : "{:.3f}".format}))
    
    df     = records_to_df(results)
    curve  = make_similarity_curve(df)
    print("\n% Same Answer curve\n", curve, "\n")
    plot_curve(curve, title="Qwen-7B Early Answer ")