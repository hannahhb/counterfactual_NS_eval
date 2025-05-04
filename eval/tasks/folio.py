"""
FOLIO: Natural Language Reasoning with First-Order Logic
https://arxiv.org/pdf/2209.00840.pdf
"""
from linc2.eval.tasks import OWAFOLTask
from linc2.eval.tasks.utils import evaluate, convert_to_nltk_rep
import json 
from pathlib import Path

_CITATION = """
@article{han2022folio,
  title={Folio: Natural language reasoning with first-order logic},
  author={Han, Simeng and Schoelkopf, Hailey and Zhao, Yilun and Qi, Zhenting and Riddell, Martin and Benson, Luke and Sun, Lucy and Zubova, Ekaterina and Qiao, Yujie and Burtell, Matthew and others},
  journal={arXiv preprint arXiv:2209.00840},
  year={2022}
}
"""




class FOLIOBase(OWAFOLTask):
    def __init__(self, model_name, model_server, mode="baseline",
                 n_shot=3, k=5, run=1):
        super().__init__(model_name, model_server, mode, n_shot, k, run,
                         dataset_type="counterfactual")
        counter_path = "data/folio_v2_perturbed.jsonl"
        self.test_dataset = self.load_jsonl_dataset(counter_path)
        self.exp_name += "_folio"
    
    
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
                for line in f:
                    if line.strip():  # Skip empty lines
                        example = json.loads(line)
                        dataset.append(example)
            
            print(f"Successfully loaded {len(dataset)} examples from {file_path}")
            return dataset
        except Exception as e:
            print(f"Error loading JSONL file: {e}")
            return []
  