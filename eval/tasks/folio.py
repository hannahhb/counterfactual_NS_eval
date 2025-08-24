"""
FOLIO: Natural Language Reasoning with First-Order Logic
https://arxiv.org/pdf/2209.00840.pdf
"""
from .owafoltask import OWAFOLTask
from .utils import evaluate, convert_to_nltk_rep
import json 
from pathlib import Path
from datasets import load_dataset
import sys


_CITATION = """
@article{han2022folio,
  title={Folio: Natural language reasoning with first-order logic},
  author={Han, Simeng and Schoelkopf, Hailey and Zhao, Yilun and Qi, Zhenting and Riddell, Martin and Benson, Luke and Sun, Lucy and Zubova, Ekaterina and Qiao, Yujie and Burtell, Matthew and others},
  journal={arXiv preprint arXiv:2209.00840},
  year={2022}
}
"""

class FOLIOBase(OWAFOLTask):
    DATASET_PATH = "benlipkin/folio"
    DATASET_NAME = None
    FULL_DATA = True

    def __init__(self, model_name, model_server, mode="baseline",
                 n_shot=3, k=5, run=1, dataset_type="counterfactual",  notes = "SAMPLE NOTE", 
                 structured = "json", do_verify = False):
        
        super().__init__(model_name, model_server, mode, n_shot, k, run,
                         dataset_type=dataset_type,  notes = notes, 
                 structured = structured, do_verify = do_verify)
        
        if self.FULL_DATA: 
            # test_dataset = self.reformat_fol_samples(self.dataset["validation"])
            # self.test_dataset = self.reformat_fol_samples(test_dataset)
            counter_path = "data/folio_counterfactual_new.jsonl"

            self.test_dataset = self.load_jsonl_dataset(counter_path)

            self.exp_name += "_folio_full"
        else:
            counter_path = "data/folio_v2_perturbed.jsonl"
            # counter_new = "data/new_perturbed.jsonl"
            self.test_dataset = self.load_jsonl_dataset(counter_path)
            self.exp_name += "_folio"
            
        
        print("Loaded test data")
        
    def reformat_fol_sample(self, sample):
        
        if self.FULL_DATA:
            sample["premises-FOL"] = [
                convert_to_nltk_rep(premise) for premise in sample["premises-FOL"]
            ]
            sample["conclusion-FOL"] = convert_to_nltk_rep(sample["conclusion-FOL"])
            
            sample["premises"] = sample["premises"]
            sample["conclusion"] = sample["conclusion"]
            # print(sample)
        else: 
            sample["orig_premises-FOL"] = [
                convert_to_nltk_rep(premise) for premise in sample["orig_premises-FOL"].split("\n")
            ]
            # sample["premises-FOL"] = [
            #     convert_to_nltk_rep(premise) for premise in sample["premises-FOL"].split("\n")
            # ]
            sample["orig_conclusion-FOL"] = convert_to_nltk_rep(sample["orig_conclusion-FOL"])
            # sample["conclusion-FOL"] = convert_to_nltk_rep(sample["conclusion-FOL"])

            sample["premises"] = sample["premises"].split("\n")
            sample["orig_premises"] = sample["orig_premises"].split("\n")
            try:
                # print( len(sample["premises"]), len(sample["premises-FOL"]) )
                assert len(sample["premises"]) == len(sample["orig_premises-FOL"])
                # print(sample["premises-FOL"], sample["orig_conclusion-FOL"])
                label = evaluate(sample["orig_premises-FOL"], sample["orig_conclusion-FOL"])
                # print("label is", label)
                
                # assert sample["label"] == label
            except Exception as e:
                print(f"Error in parsing FOL: {e}")
                # print(sample)
            
            # sample["label"] = label 
            # sample["error"] = True
        # sample["error"] = False
        # print(sample)
        return sample
        
    def load_jsonl_dataset(self, file_path):
        """
        Load a JSONL file directly without using the datasets library.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries containing the dataset examples
        """
        dataset = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Warning: File {file_path} not found")
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                error_count=0
                for idx, line in enumerate(f):
                    if line.strip():  # Skip empty lines
                        example = json.loads(line)
                        example = self.reformat_fol_sample(example)
                        if example["label"] == self.ERROR_TOKEN:
                            error_count+=1
                        
                        dataset.append(example)
                    # if idx ==1:
                    #     sys.exit("STOP EXEC TESTING")
                print("Num errors in ds: ", error_count)
            
            print(f"Successfully loaded {len(dataset)} examples from {file_path}")
            return dataset
        except Exception as e:
            print(f"Error loading JSONL file: {e}")
            return []
        
    def reformat_fol_samples(self, dataset):
        
        data = []
        # print(len(dataset))
        count_error=0 
        for idx in range(len(dataset)):
            sample = self.reformat_fol_sample(dataset[idx])
            if sample["label"] == self.ERROR_TOKEN:
                count_error+=1
            data.append(sample)
            # if idx ==1:
            #     sys.exit("STOP EXEC TESTING")
        print("Num_errors in test data = ", count_error)
        # print(len(data))
        
        return data