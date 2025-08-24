from collections import Counter
# counterfactual_maker.py
import json, random, time, re, sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
                    
sys.path.insert(0, "/data/projects/punim0478/bansaab/linc2/")

from eval.server import ModelServer
from eval.tasks import FOLIOBase

def find_boundary_samples(progress_json_path, top_n=None):
    with open(progress_json_path, 'r', encoding='utf-8') as f:
        progress = json.load(f)

    boundaries = []
    for idx, record in progress.items():
        gens = record.get('generations', [])
        counts = Counter(gens)
        if not counts:
            continue
        most_common = counts.most_common(2)
        top_count = most_common[0][1]
        runner_up_count = most_common[1][1] if len(most_common) > 1 else 0
        margin = top_count - runner_up_count
        boundaries.append((idx, margin, counts))

    boundaries.sort(key=lambda x: x[1])
    return boundaries[:top_n] if top_n else boundaries


    

_BAD_JSON_RE = re.compile(r'```json|```', re.I)

class CounterfactualMaker:
    """Query an LLM to rewrite 3-4 salient nouns in every FOLIO example."""
    def __init__(self,
                 model_server,              # your ModelServer instance
                 shots: int = 0,            # few-shots if you like
                 n_swaps: int = 4,
                 seed: int = 0):
        self.llm   = model_server
        self.n_swaps = n_swaps
        self.rng   = random.Random(seed)
        self.few_shot_header = (
            "Task: Replace exactly THREE to FOUR nouns/"
            "proper names/places/attributes in the example with wrong but"
            "grammatically valid tokens.  Keep *logical structure* and `label`"
            "unchanged.  Update only natural-language text, no need for the FOL. \n"
            "Return only valid premises and conclusions in line seperated format. For example as replace 'people' with 'animals', 'Taylor Swift' with 'Madonna' or something like that. If you change an instance in one place you need to change it in all sentences."
            "If there are multiple proper nouns definitely change atleast 1 so it becomes infactual. If there are celibrity names, city names or anything that is unique in human knowledge definitely swap atleast one out. Like Ailton, Beijing, Square, Taylor Swift\n\n"
        )

    # ------------------------------------------------------------------ #
    def _craft_prompt(self, ex: Dict[str, Any]) -> str:
        """One prompt per example."""
        stub = json.dumps({
            "premises": ex["premises"],
            "premises-FOL": ex["premises-FOL"],
            "conclusion": ex["conclusion"],
            "conclusion-FOL": ex["conclusion-FOL"],
            "label": ex["label"]
        }, ensure_ascii=False)
        return (
            self.few_shot_header
            # + "EXAMPLE:\n"
            # + """{"premises": ["If people perform in school talent shows often, then they attend and are very engaged with school events.", "People either perform in school talent shows often or are inactive and disinterested members of their community.", "If people chaperone high school dances, then they are not students who attend the school.", "All people who are inactive and disinterested members of their community chaperone high school dances.", "All young children and teenagers who wish to further their academic careers and educational opportunities are students who attend the school.", "Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school. "], "premises-FOL": ["\u2200x (TalentShows(x) \u2192 Engaged(x))", "\u2200x (TalentShows(x) \u2228 Inactive(x))", "\u2200x (Chaperone(x) \u2192 \u00acStudents(x))", "\u2200x (Inactive(x) \u2192 Chaperone(x))", "\u2200x (AcademicCareer(x) \u2192 Students(x))", "(Engaged(bonnie) \u2227 Students(bonnie)) \u2295 (\u00acEngaged(bonnie) \u2227 \u00acStudents(bonnie))"], "conclusion": "Bonnie performs in school talent shows often.", "conclusion-FOL": "Engaged(bonnie)", "label": "Uncertain"}"""
            # + "\n\nREWRITE:\n"
            # + "If animals perform in school talent shows often, then they attend and are very engaged with school events.\n"\
            # "Animals either perform in school talent shows often or are inactive and disinterested members of their community.\n"\
            # "If animals chaperone high school dances, then they are not puppies who attend the school.\n"\
            # "All animals who are inactive and disinterested members of their community chaperone high school dances.\n"\
            # "All young cattle and teenagers who wish to further their academic careers and educational opportunities are puppies who attend the school.\n"\
            # "Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school. \n"\
            # "Bonnie performs in school talent shows often."
            + "\n\nEXAMPLE:\n"
            +"""{"premises": ["All squares have four sides.", "All four-sided things are shapes. "], "premises-FOL": ["\u2200x (Square(x) \u2192 FourSides(x))", "\u2200x (FourSides(x) \u2192 IsShape(x))"], "conclusion": "All squares are shapes.", "conclusion-FOL": "\u2200x (Square(x) \u2192 IsShape(x))", "label": "True"}"""
            + "\n\nREWRITE:\n"\
            "All triangles have four sides.\n"\
            "All four-sided things are potatoes.\n"\
            "All triangles are potatoes."
            + "\n\nEXAMPLE:\n"
            + stub
            + "\n\nREWRITE:\n"
        )

    # ------------------------------------------------------------------ #
    def _call_llm(self, prompt: str, max_retries=3) -> Dict[str, Any]:
        params = dict(temperature=0.7, max_tokens=800)
        for attempt in range(max_retries):
            out = self.llm.generate(prompt, params)
            return out
            # try:
            #     obj = json.loads(out)
            #     # minimal sanity checks
            #     for k in ["premises", "premises-FOL",
            #               "conclusion", "conclusion-FOL", "label"]:
            #         assert k in obj
            #     return obj
            # except Exception as e:
            #     print(f"[warn] JSON parse fail #{attempt+1}: {e}")
            #     time.sleep(1.5 * (attempt + 1))
        raise ValueError("LLM failed to return valid JSON 3×")

    # ------------------------------------------------------------------ #
    def make_cf_set(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cf_examples = []
        for idx, ex in tqdm(enumerate(dataset),
                            total=len(dataset), desc="Generating CF"):
            # if idx==24:
                prompt = self._craft_prompt(ex)
                print(prompt)
                cf = self._call_llm(prompt)
                # print(cf)
                chunks = [s.strip() for s in re.split(r'\n', cf.strip()) if s.strip()]
                premises    = chunks[:-1]   # first n-1
                conclusion  = chunks[-1]    # last one
                
                stub = json.dumps({
                    "premises": ex["premises"],
                    "premises-FOL": ex["premises-FOL"],
                    "conclusion": ex["conclusion"],
                    "conclusion-FOL": ex["conclusion-FOL"],
                    "label": ex["label"],
                    "premises_nouns_p": premises, 
                    "conclusion_nouns_p": conclusion
                }, ensure_ascii=False)
                # print(stub)
                cf_examples.append(stub)
        return cf_examples


if __name__ == "__main__":
    gemini_flash_2_5 = "google/gemini-2.5-flash-preview-05-20"

    vllm_server = ModelServer(
        model_name=gemini_flash_2_5,
        mode="openrouter",
        n_gpu=1
    )
    
    task = FOLIOBase(
        mode="baseline",
        model_server=vllm_server, 
        model_name=gemini_flash_2_5,
        n_shot=8, 
        k=10,
        dataset_type="default",
    )
    orig_data = task.test_dataset
    cf_maker = CounterfactualMaker(vllm_server)
    cf_data  = cf_maker.make_cf_set(orig_data)
    with open("folio_counterfactual_new.jsonl", "w", encoding="utf-8") as f:
        for ex in cf_data:
            f.write(ex)
# if __name__ == "__main__":
#     # ... unchanged printing of boundary idxs ...
#     progress_file = sys.argv[1]
#     top_n = int(sys.argv[2]) if len(sys.argv)>2 else None
#     bs = find_boundary_samples(progress_file, top_n)
#     print("Idx\tMargin\tVoteCounts")
#     print("[", ", ".join(str(idx) for idx,_,_ in bs), "]")
    