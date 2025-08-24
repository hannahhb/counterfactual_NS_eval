from .owafoltask import OWAFOLTask
from datasets import load_dataset

_CITATION = """
@inproceedings{Tafjord2020ProofWriterGI,
  title={ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language},
  author={Oyvind Tafjord and Bhavana Dalvi and Peter Clark},
  booktitle={Findings},
  year={2020}
}
"""

class ProofWriterTask(OWAFOLTask):
    """Use ProofWriter’s balanced deduction test split as the test set."""
    PROOFWRITER_HF = "theoxo/proofwriter-deduction-balanced"

    def __init__(self, model_name, model_server, mode="baseline",
                 n_shot=3, k=5, run=1, seed=7):
        super().__init__(model_name, model_server, mode, n_shot, k, run,
                         dataset_type="default")
        pw = load_dataset(self.PROOFWRITER_HF, split="test").select(range(100))
        # apply your reformat_proofwriter helper
        self.test_dataset = self.reformat_proofwriter(pw).shuffle(seed)
        self.exp_name += "_proofwriter"      # optional suffix

    def reformat_proofwriter(self, dataset):

        def punctuate(s):
            if s[-1] not in [".", "?", "!"]:
                s += "."
            return s

        def reformat_sample(sample):
            sample["premises"] = [punctuate(p) for p in sample.pop("theory").split(". ")]
            sample["conclusion"] = punctuate(sample.pop("question"))
            sample["label"] = sample.pop("answer")
            return sample

        return dataset.map(reformat_sample)
    
    def format_test_example(self, doc, is_default = True):
        """Fixed test example formatting"""
        example = "<PREMISES>\n"
        # Properly split multi-line premises
        prem = "premises"
        concl = 'conclusion'
        
        premises = doc[prem]
        # print(premises)
        
        if isinstance(premises, list):
            # already a list of sentences
            premises = premises
        else:
            # split a multiline string into lines, drop blank lines
            premises = [line.strip() for line in premises.splitlines() if line.strip()]

        for premise in premises:
            example += f"{premise}"  
            example += "\n"
      
        example += "</PREMISES>\n"
        example += f"<CONCLUSION>\n{doc[concl].strip()}\n</CONCLUSION>\n"
        example += "<EVALUATE>\n"
        return example