import json
from pathlib import Path
from tqdm import tqdm

# 1) small helper to extract nouns via spaCy (so we can seed Gemini with alternatives)
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_nouns(text):
    doc = nlp(text)
    return [tok.text for tok in doc if tok.pos_ == "NOUN"]

# 2) build the prompt for Gemini
def build_perturb_prompt(sample):
    """
    Given one FOLIO example with sample['orig_premises'] (List[str]) 
    and sample['orig_conclusion'] (str), produce a single prompt
    instructing the model to swap nouns and add a confusing premise.
    """
    premises = sample["orig_premises"]
    conclusion = sample["orig_conclusion"]

    # gather nouns from all premises/conclusion to suggest swapping
    all_nouns = set()
    for sent in premises + [conclusion]:
        all_nouns.update(extract_nouns(sent))
    # pick two to swap in the prompt
    nouns = list(all_nouns)
    if len(nouns) >= 2:
        a, b = nouns[0], nouns[1]
    else:
        # fallback to generic placeholders
        a, b = "Alice", "Bob"

    prompt = f"""
You are going to make a _perturbed_ version of the following FOLIO example by:
  1) Swapping every occurrence of the noun “{a}” with “{b}” (and vice versa)
  2) Inserting one extra “confusing” premise that looks _syntactically_ like the others
     (for example, repeating a similar template but using the new noun)
  3) Keeping the exact same sentence structure, punctuation, and number of lines

Original premises:
{"    • ".join(premises)}

Original conclusion:
    {conclusion}

Return your new premises (same count), then the new conclusion, in JSON format:

{{
  "orig_id": {sample["idx"]!r},
  "perturbed_premises": [ /* list of strings, same length */ ],
  "perturbed_conclusion": /* string */
}}
"""
    return prompt.strip()

# 3) parse responses back into our format
def parse_perturbation_output(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # crude fallback: try to grab {...} substring
        import re
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))

# 4) the pipeline
def make_perturbed_dataset(model_server, dataset, out_path):
    """
    model_server: your ModelServer("google/gemini-2.0-flash-exp:free", mode="server", ...)
    dataset:      list of samples loaded by your FOLIOBase
    out_path:     path to write the new JSONL
    """
    params = {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.95,
        # no stop words here – we want complete JSON
    }

    with open(out_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(dataset, total=len(dataset)):
            prompt = build_perturb_prompt(sample)
            # rate‐limit if needed, e.g. for Gemini only
            pert_text = model_server.generate(prompt, params)
            pert = parse_perturbation_output(pert_text)

            # sanity‐check: same number of premises
            assert len(pert["perturbed_premises"]) == len(sample["orig_premises"])

            # emit a JSONL line combining everything
            out = {
                "idx": sample["idx"],
                "orig_premises": sample["orig_premises"],
                "orig_conclusion": sample["orig_conclusion"],
                "perturbed_premises": pert["perturbed_premises"],
                "perturbed_conclusion": pert["perturbed_conclusion"],
                "label": sample["label"],
            }
            fout.write(json.dumps(out) + "\n")

    print(f"Wrote perturbed dataset → {out_path}")
