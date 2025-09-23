# Robustness of Neurosymbolic Reasoners on First-Order Logic Problems

This repository provides a simple entry point for running experiments using **LINC**, **NSCoT**, **CoT**, **Scratchpad**, and **Naïve** methods on **FOLIO Default** and **RR**. 

All experiments are run through a single script: `main.py`.  
You can directly configure **model**, **dataset**, and **mode** inside `main.py` to explore different experimental settings.

---

## Quick Start

Run experiments with:

```bash
python main.py
```

The following parameters are changable in main.py:

```python 
model_name = ["qwen-7b"] # choose from the models in the given file 
dataset = ["default"] # default or counterfactual
mode = ["cot"] # baseline, scratchpad, neurocot (NSCoT), or neurosymbolic (linc)
```

- Results and logs will be written to the output directory specified in the code (`results/model_name/mode`).
- Multiple generations with majority voting are used for robustness in all methods. You can change K (=number of generations) and shots in `main.py` as well.

## Installation

Install dependencies with:
```bash
pip install -r requirements.txt
```

[Prover9](https://www.cs.unm.edu/~mccune/mace4/) is required for LINC/NSCoT runs (ensure it is installed and available in your PATH).
