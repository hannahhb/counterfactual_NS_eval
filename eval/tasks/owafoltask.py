from functools import cache
from collections import Counter
from .utils import evaluate, convert_to_nltk_rep
from abc import abstractmethod, ABC
from datasets import load_dataset
import re
from warnings import warn

from pydantic import BaseModel, Field
from typing import List

from concurrent.futures  import ThreadPoolExecutor
from vllm.sampling_params import GuidedDecodingParams
import os 
from tqdm import tqdm 
import json
from pathlib import Path
import time
from collections import defaultdict


from .tasks_base import Task

class OWAFOLTask(Task):
    """A First-Order Logic Inference Task following the Open World Assumption."""
    TRAIN_DATASET_PATH = "minimario/FOLIO"
    TRAIN_DATASET_NAME = "train"
    MODES = ["baseline", "cot", "scratchpad", "neurosymbolic", "neurocot", "neurostep"]
    ERROR_TOKEN = "Error"
    MAX_SHOTS = 16
    DEBUG = False
    # verify = False
    NUM_VERIFY = 2
    
    len_test_lines  = 0
    port = 8000
    
    def __init__(self, model_name, model_server,  mode = "baseline", n_shot=3, k=5, run=1, 
                 dataset_type = "counterfactual", notes = "SAMPLE NOTE", 
                 structured = "json", do_verify = False):
        """
        :param mode: str
            Inference mode. One of ["baseline", "cot", "scratchpad", "neurosymbolic"]
        :param n_shot: int
            Number of few-shot examples to use
        :param model_name: str
            HuggingFace model name
        """
        assert mode in self.MODES, f"Invalid mode. Choose from {self.MODES}"
        assert n_shot <= self.MAX_SHOTS, f"Supports up to {self.MAX_SHOTS}-shot"
        stop_words = ["</EVALUATE>"] 
        
        REASONING_CHAIN_PATH = "/data/projects/punim0478/bansaab/linc2/data/reasoning_examples.json"
        with open(REASONING_CHAIN_PATH) as f:
            self.fols_cot = json.load(f)

        if structured:
            stop_words.append("python")
        super().__init__(
            stop_words=stop_words, requires_execution=(mode == "neurosymbolic"),
        )
        
        
        # self.test_data_name = test_data
        # counter_path = "data/folio_v2_perturbed.jsonl"
            
        self.mode = mode
        self.n_shot = n_shot
        self.model_name = model_name
        self.k = k
        
        self.dataset_type = dataset_type
        self.train_dataset = self.prepare_train_dataset()
        
        base_exp_name = f"{model_name}/{mode}_k{k}"
        self.exp_name = f"{base_exp_name}_run{run}_s{self.n_shot}_{dataset_type}"
        self.model_server = model_server
        self.notes = notes
        
        self.do_verify = do_verify
        self.structured = structured
        # self.step_wise = step_wise
        self._last_arity_map = None
        
    def prepare_train_dataset(self):
        """Prepares the training dataset with necessary preprocessing."""
        train_dataset = load_dataset(self.TRAIN_DATASET_PATH, split='train')
        # train_dataset.cleanup_cache_files()

        # Reformat premises to NLTK representations
        train_dataset = train_dataset.map(self.reformat_fol_samples_train,  num_proc=1, load_from_cache_file=False)

        # Add conclusion FOL expressions
        train_dataset = self.add_conclusion_fols_train(train_dataset)

        # Add Chain-of-Thought (CoT) explanations
        train_dataset = self.add_cot_train(train_dataset)
        
        # Add Chain-of-Thought (CoT) FOL explanations
        train_dataset = self.add_reasoning_chains(train_dataset, self.fols_cot)
        
        # Map labels to standardized format
        train_dataset = train_dataset.map(
            lambda x: {"label": "Uncertain" if x["label"] == "Unknown" else x["label"]},
            remove_columns=["label"],
        )

        # Select few-shot examples based on predefined indices
        fewshot_indices_all = [
            125, 23, 60, 275, 148, 261, 263, 683, 299, 684, 850, 853, 886, 892, 930, 980,
        ]
        fewshot_indices = fewshot_indices_all[:self.n_shot]
        train_fewshot = train_dataset.select(fewshot_indices)
        return train_fewshot

    def reformat_fol_samples_train(self, sample):
        """
        Takes one example (LazyRow) whose 'premises-FOL' is a list of
        Unicode FOL strings like ["∀x (P(x) → Q(x))", "(A ∧ B) ⊕ C", …].
        Replace each with its ASCII/NLTK equivalent.
        """
        raw_fols = sample["premises-FOL"]
        # Ensure we have a list
        if isinstance(raw_fols, str):
            # fallback if something slipped through as a newline string
            raw_fols = [l.strip() for l in raw_fols.split("\n") if l.strip()]

        # print(raw_fols)
        # Apply your converter to each one
        sample["premises-FOL"] = [
            convert_to_nltk_rep(fol) 
            for fol in raw_fols
        ]
        
        # print(sample)
        return sample

    
    def format_test_example(self, doc, train=False):
        """Fixed test example formatting, works if doc[premises] is str or List[str]."""
        example = "<PREMISES>\n"
        # print(doc)
        if self.dataset_type == "counterfactual": 
            key_prem  = "premises" 
            key_concl = "conclusion" 
        elif self.dataset_type == "default": 
            key_prem  = "orig_premises"
            key_concl = "orig_conclusion"
            if train or self.FULL_DATA == True: 
                key_prem = "premises"
                key_concl = "conclusion"
            
        # print(doc)
        raw_prem = doc[key_prem]
        # unify to a list of lines
        if isinstance(raw_prem, list):
            premises = [p.strip() for p in raw_prem if p and p.strip()]
        else:
            premises = [p.strip() for p in raw_prem.split("\n") if p.strip()]

        for p in premises:
            example += f"{p}\n"
        self.len_test_lines = len(premises) + 1
        example += "</PREMISES>\n"
        example += f"<CONCLUSION>\n{doc[key_concl].strip()}\n</CONCLUSION>\n"
        example += "<EVALUATE>\n"
        # print(example)
        return example
    
    def format_train_example(self, doc, stage = 2):
        """Fixed formatting for training examples"""
        example = self.format_test_example(doc, train=True)
        
        # Handle premises properly
        if self.mode == "baseline":
            example += f"ANSWER: {doc['label'].strip()}\n"
        elif self.mode == "cot":
            example += f"\n{doc['cot']}\n"
        else:
            premises = doc["premises"]
            
            all_raw = []
            example_neuro = ""
            
            steps = doc.get("cot_fol", [])
            # print(steps)
            for i, (premise, fol) in enumerate(zip(premises, doc["premises-FOL"])):
                example_neuro += f"TEXT: {premise}\n"
                
                if self.mode == "neurocot":
                    example_neuro += f"REASONING: {steps[i]}\n"
                    # print(steps[i])
                    
                example_neuro += f"FOL: {fol}\n"
                # print(fol)
                all_raw.append(fol)
            
            conclusion_fol = doc['conclusion-FOL'].strip()
            all_raw.append(conclusion_fol)
            
            example_neuro += f"TEXT: {doc['conclusion'].strip()}\n"
            
            if self.mode == "neurocot":
                example_neuro += f"REASONING: {steps[-1]}\n"
            
            example_neuro += f"FOL: {conclusion_fol}\n"
            
            signature_lines = self._extract_signature(all_raw)
            if self.mode == "neurostep":
                example += f"Predicate/arity map for the problem: \n {signature_lines}"
                if stage==1:
                    return example
          
            example+= f"{example_neuro}"
            # print(signature_lines)
            
            if self.mode == "scratchpad":
                example += f"\nANSWER: {doc['label'].strip()}\n"
        
        return example + "</EVALUATE>\n"
    
    
    def add_conclusion_fols_train(self, train_dataset):
        """Adds conclusion FOL expressions to the training dataset."""
        train_conclusion_fols = {
            23: "HigherRank(RealMadrid, Barcelona)",
            60: "-OlympicGoldMedalWinner(Amy) -> NobelLaureate(Amy)",
            125: "-Dispensable(Worksheet)",
            148: "FolkSong(Inception)",
            261: "MakeGoodBreakfast(Luke)",
            263: "exists x. (Develops(Ets, x) & For(x, k-OneTwoandhighereducation)) & exists x. (Develops(Ets, x) & AssociatedWith(x, Entrytouseducationinstitutions))",
            275: "ContributeToCountry(James)",
            299: "GetRhythmRight(John)",
            683: "exists x. (BRICS(x) & Speak(x, Hindi))",
            684: "Film(Hamilton)",
            850: "-Liked(Leo, Charlie) & -Cares(Charlie, Leo)",
            853: "Won(Threebodyproblem, Hugoaward)",
            886: "Dagfinn(DagfinnAarskog)",
            892: "PartOf(Minsk, Scottishpremiership)",
            930: "-Locate(Boves, Europe)",
            980: "(InvitedTakePhoto(James) & -HappyCommunicate(James)) | (-InvitedTakePhoto(James) & HappyCommunicate(James))",
        }
        conclusions = [None for _ in range(len(train_dataset))]
        for index, conclusion_fol in train_conclusion_fols.items():
            if index < len(conclusions):
                conclusions[index] = conclusion_fol

        # Remove the column if it exists
        if "conclusion-FOL" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns("conclusion-FOL")
        
        # Add the new column
        train_dataset = train_dataset.add_column("conclusion-FOL", conclusions)
        return train_dataset
    
    def add_cot_train(self, train_dataset):
        """Adds Chain-of-Thought explanations to the training dataset."""
        train_cots = {
            23: "Let's think step by step. We want to evaluate if in La Liga 2021-2022, Real Madrid ranks higher than Barcelona. From premise 1, we know that a La Liga soccer team ranks higher than another if it receives more points. From premise 4, we know that in La Liga 2021-2022, Real Madrid received more points than Barcelona. Therefore, in La Liga 2021-2022, Real Madrid received more points than Barcelona, so Real Madrid ranks higher than Barcelona, so the statement is true.\nANSWER:\tTrue",
            60: "Let's think step by step. We want to evaluate the statement \"if Amy is not an Olympic gold medal winner, then Amy is a Nobel laureate\". Let's assume that Amy is not an Olympic gold medal winner. This doesn't tell us anything about whether Amy is a Nobel laureate, so the statement isn't true, meaning it is either False or Uncertain. To distinguish between the two, notice that we could have a scenario where Amy is neither an Olympic gold medal winner nor a Nobel laureate. None of the premises are violated in this case. This means the statement must be false.\nANSWER:\tFalse",
            125: "Let's think step by step. We want to evaluate if a worksheet is not dispensable. From premise 6, we know that a worksheet is either paper or is environment-friendly. If it is paper, then from premise 3, a worksheet is woodware, and from premise 2, a worksheet is dispensable. If it is environment-friendly, we know it is good from premise 5, but we know nothing about whether it is dispensable. Therefore, we don't know if a worksheet is dispensible or not, so the statement is uncertain.\nANSWER:\tUncertain",
            148: "Let's think step by step. We want to evaluate if Inception is a folk song. We know that Inception is a sci-fi movie. Since all movies are videos and Inception is a movie, it is a video, which means it is visual. On the other hand, we know that all folk songs are songs, and no songs are visual, so no folk songs are visual. Therefore, since Inception is visual but no folk songs are visual, we know that Inception cannot be a folk song, so the statement is false.\nANSWER:\tFalse",
            261: "Let's think step by step. We want to evaluate if Luke can make a good breakfast. From the last premise, we know that Luke can make cookies, scrambled eggs, and muffins. Since Luke can make cookies and muffins, they are a baker. Now, combining the information we have, since Luke is a baker and can make scrambled eggs, this means that they can make a good breakfast. Therefore, Luke can make a good breakfast, so the statement is true.\nANSWER:\tTrue",
            263: "Let's think step by step. We want to evaluate if ETS develops assessments for K-12 statewide as well as entry to US tertiary and quaternary educatiand doon institutions. We know that ETS develops assessments for K-12 statewide. We also know that ETS develops assessments associated with entry to the US tertiary and quaternary education institutes. Therefore, both parts of the conclusion are true, and the statement is true.\nANSWER:\tTrue",
            275: "Let's think step by step. We want to evaluate if James contributes to the country. Let's think about what we know about James. First, we know that James was either sentenced for thief or stayed in prison. However, this doesn't tell us anything about whether James contributed to the country. Second, we know that James either had a bad record in the local state or that he was respected by others. However, the premises don't tell us anything about the relationship between having a bad record and contributing to the country. Therefore, it is uncertain whether James contributes to the country.\nANSWER:\tUncertain",
            299: "Let's think step by step. We want to evaluate if John can get the rhythms right. We know that John is a student learning piano. Since all students learning piano can strike the right notes, John can strike the right notes. Since all students who can strike the right notes can get the rhythms right and John can strike the right notes, John can get the rhythms right, so the conclusion is true.\nANSWER:\tTrue",
            683: "Let's think step by step. We want to evaluate if there is a person from BRICS speaking Hindi. We know that there is an Indian, and since India is one of BRICS, we know that there is an Indian in BRICS. Furthermore, we know that they speak either Hindi or English, however, we don't know which one. Therefore, there could be a person in BRICS speaking Hindi, or there could not. Therefore, it is uncertain whether there is a person from BRICS speaking Hindi.\nANSWER:\tUncertain",
            684: "Let's think step by step. We want to evaluate if Hamilton is a film. Since Daveed Diggs played two roles in the musical Hamilton, Hamilton is a musical. Since musicals are not films and Hamilton is a musical, Hamilton is not a film, and the conclusion is false.\nANSWER:\tFalse",
            850: "Let's think step by step. We want to evaluate if Charlie does not like Leo and does not care for Leo. Let's first evaluate if Charlie does not like Leo. We know Charlie has a naughty pet named Leo. Since pets who are naughty are not liked as much, Charlie does not like Leo. Now, let's evaluate if Charlie cares for Leo. We know that if a person has a pet, they care for that pet. Since Leo is Charlie's pet, Charlie cares for Leo. Therefore, Charlie does not like Leo but cares for Leo, so the second part of the conclusion is false, which means the entire conclusion is false.\nANSWER:\tFalse",
            853: "Let's think step by step. We want to evaluate if the Three Body Problem won the Hugo Award. The only thing we know about the Hugo Award is that some books that have won the Hugo Award were written by Cixin Liu. However, we know nothing about whether The Three Body Problem was written by Cixin Liu, so the conclusion is uncertain.\nANSWER:\tUncertain",
            886: "Let's think step by step. We want to evaluate if Dagfinn is Dagfinn Aarskog's given name. We know that Dagfinn is a given name, and that notable people with the given name Dagfinn includes Dagfinn Aarskog, which means that Dagfinn is Dagfinn Aarskog's given name, so the conclusion is true.\nANSWER:\tTrue",
            892: "Let's think step by step. We want to evaluate if Minsk joined the Scottish Premiership. We know that Minsk and St Johnstone are different teams and that St Johnstone is part of the Scottish Premiership, but we don't know anything about whether or not Minsk joined the Scottish Premiership from the premises. Therefore, the conclusion is uncertain.\nANSWER:\tUncertain",
            930: "Let's think step by step. We want to evaluate if Boves is not in Europe. We know that Boves is a railway station located in France. We also know that since France is a European country, France is located in Europe. Furthermore, we know that if A is located in B and B is located in C, then A is located in C. Therefore, we know that because Boves is located in France and France is located in Europe, that means Boves is located in Europe. Therefore, the conclusion is false.\nANSWER:\tFalse",
            980: "Let's think step by step. We want to evaluate if James is either invited to take a photo with the audience or happy to communicate with each other during the dinner. We know that James does not attend the conference in person and is not provided with souvenirs. There are no premises that apply to people who do not attend the conference. Since James is not provided with souvenirs, since all who attended the conference in person are provided with souvenirs, we know that James did not attend the conference in person. However, we don't know anything else, so it is possible that James was neither invited to take a photo with the audience nor happy to communicate during the dinner. Therefore, the conclusion is false.\nANSWER:\tFalse",
        }
        cots = [None for _ in range(len(train_dataset))]
        for index, cot in train_cots.items():
            if index < len(cots):
                cots[index] = cot
                
        train_dataset = train_dataset.add_column("cot", cots)
        return train_dataset
    
    
    def add_reasoning_chains(
        self,
        train_dataset,
        chains: dict[int, list[str]],
        column_name: str = "cot_fol",
    ):
        """
        Given a HF Dataset (or any list-like of examples) and a mapping from
        example index to a list of reasoning steps, return a new Dataset
        with a column `column_name` where each row i gets chains[i] (a list)
        or None if missing.
        """
        n = len(train_dataset)
        out: list[list[str] | None] = [None] * n

        for idx_str, steps in chains.items():
            idx = int(idx_str)
            # store the raw list of steps
            out[idx] = steps

        return train_dataset.add_column(column_name, out)

    # @cache
    def fewshot_examples(self, stage: int = 2):
        examples = [
            self.format_train_example(doc, stage=stage)
            for doc in self.train_dataset
        ]
        # separate each example with a delimiter so the LLM sees them distinctly
        return "\n---\n".join(examples) + "\n---\n"


    def get_prompt_stage1(self, doc):
        instr = self.get_instructions(stage=1) + \
                "Stage 1: list your PREDICATE/ARITY map only in format []. Don't make the predicate names too unneceraily long and break them up where necessary." \
                "Don't make predicates that are subsets of others such as EggLayingMammal and EggLaying. Don't print any extra reasoning \n\n "
        train  = self.fewshot_examples(stage=1)
        test   = self.format_test_example(doc)
        return instr + train + test

    def get_prompt_stage2(self, doc, raw_arity_map):
        instr = self.get_instructions()
        train  = self.fewshot_examples()
        test   = self.format_test_example(doc)  + \
                f"{raw_arity_map} \n\n"
        
        return instr + train + test


    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """

        instructions = self.get_instructions()
        # self.fewshot_examples.cache_clear()

        train = self.fewshot_examples()
        test = self.format_test_example(doc)
        prompt = "\n".join([instructions, train, test])
        if self.mode == "neurostep":
            # --- Stage 1: get the predicate/arity map ---
            stage1 = self.get_prompt_stage1(doc)
            print(stage1)
            
            _map_raw = self.generate_with_vllm(stage1, stage=1)
            print("-----PRED/ARITY MAP------")
            print(_map_raw)
            # --- Stage 2: generate TEXT/FOL pairs re-using that map ---
            prompt = self.get_prompt_stage2(doc, _map_raw)
        # else:
            # old single-stage
        return prompt

    
   
    def _extract_signature(self, docs):
        """
        Given a list of raw FOL strings `docs`, returns a sorted list of 
        "PredicateName/arity" entries, including zero-arity constants.
        """
        sig = defaultdict(set)

        # Pattern to catch Name(arg1,…,argN)
        pred_pat = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s*\(([^)]*)\)')
        # Pattern to catch bare Name (zero-arity)
        const_pat = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\b(?!\s*\()')

        for doc in docs:
            if not isinstance(doc, str):
                continue

            # 1) Capture all parenthesized uses
            for name, args in pred_pat.findall(doc):
                args = args.strip()
                if args == "":
                    sig[name].add(0)
                else:
                    # split on commas, ignore empty
                    count = len([a for a in re.split(r'\s*,\s*', args) if a])
                    sig[name].add(count)

            # 2) Capture any bare constants/predicates as zero-arity
            for name in const_pat.findall(doc):
                sig[name].add(0)

        # 3) Flatten to "Name/arity", sorted
        lines = []
        for name in sorted(sig):
            for ar in sorted(sig[name]):
                lines.append(f"{name}/{ar}")

        return lines

    def make_fol_schema(self, n_lines: int):
        class FOLPair(BaseModel):
            text: str
            fol: str
        
        class FOLBatch(BaseModel):
            fol_pairs: List[FOLPair] = Field(..., min_items=n_lines, max_items=n_lines)
        return FOLBatch.model_json_schema()
    
    def make_fol_regex(self, n_lines):
        if n_lines < 1:
            raise ValueError("n must be >= 1")

        # It *does not* understand ^ or $ anchors.
        pair   = r"TEXT:\s*(.*?)\s*FOL:\s*(.*?)"   # 2 capture groups
        regex  = r"\s*".join([pair] * n_lines)           # repeat n times separated by whitespace

        return regex

    def get_instructions(self, stage=2):
        """Generates task-specific instructions based on the mode."""
        instructions = ""
        instructions += "The following is a first-order logic (FOL) problem.\n"
        instructions += "The problem is to determine whether the conclusion follows from the premises.\n"
        instructions += "The premises are given in the form of a set of first-order logic sentences.\n"
        instructions += "The conclusion is given in the form of a single first-order logic sentence.\n"
        if self.mode == "baseline":
            instructions += "The task is to evaluate the conclusion as ANSWER: True, ANSWER: False, or ANSWER: Uncertain given the premises."
        elif self.mode == "cot":
            instructions += "The task is to evaluate the conclusion as ANSWER: True, ANSWER: False, or ANSWER: Uncertain given the premises. Think step by step about your reasoning."
        else:
            
            instructions += "The task is to translate each of the premises and conclusions into FOL expressions "
            
            if self.mode == "scratchpad":
                instructions += "and then to evaluate the conclusion as ANSWER: True, ANSWER: False, or ANSWER: Uncertain given the premises." 
            elif self.mode in ["neurosymbolic", "neurocot", "neurostep"]:
                instructions += "so that the expressions can be evaluated by a theorem solver like Prover9 to determine whether the conclusion follows from the premises.\n"
                if self.mode=="neurocot":
                    instructions += "There should be a reasoning step in the middle starting with REASONING:.. \n" 
                    instructions += "The output should follow format TEXT: \nREASONING: \nFOL: \nTEXT: \nREASONING: \nFOL:..."
                elif self.mode in ["neurosymbolic", "neurostep"]:
                    if stage==2: 
                        instructions += "The output should follow format TEXT: \nFOL: \nTEXT: \nFOL:.... "
                instructions += "Expressions should adhere to the format of the Python NLTK package logic module." 
            
                # instructions += "Provide your reasoning step by step and conclude with 'ANSWER:\tTrue', 'ANSWER:\tFalse', or 'ANSWER:\tUncertain'."
                instructions += "Remember to never the FOL predicates too long or overlapping meanings with other predicates.\n " 
                instructions += "Also you must not have empty predicates like Conductor()."
                instructions += "Symbols like <, >, =, <-> are not allowed."
    
        return instructions + "\n\n"
    
    def generate_with_vllm(self, prompt, stage=2):
        """Generates text from the vLLM using the OpenAI client."""
        
        try:
            params = {
                "temperature": 0.8,
                "max_tokens": 8192, 
                "top_p": 0.92, 
                "stop": self.stop_words
            }
            
            if stage == 1:
                regex = "Predicate\/arity map for the problem:\s*\[\s*'(?:[A-Za-z_]\w*\/\d+)'(?:\s*,\s*'(?:[A-Za-z_]\w*\/\d+)')*\s*\]"
                if self.model_server.mode == "server":
                    params["extra_body"] = {
                        "guided_regex": regex,
                    }
                else:
                    params["guided_decoding"] = GuidedDecodingParams(
                        regex=regex
                    )
                        
            if self.structured != False and stage==2:
                if self.mode in ["neurosymbolic", "neurostep"]:
                    # regex = r"TEXT:\s*(.*?)\s*FOL:\s*(.*?)(?=(?:\n+TEXT:|\Z))"
                    regex = r"TEXT:\s*(.+?)\s*FOL:\s*([\s\S]+)"  
                    json_schema = self.make_fol_schema(self.len_test_lines)
                if self.mode == "neurocot":
                    regex = r"TEXT:\s*(.+?)\s*REASONING:\s*(.+?)\s*FOL:\s*([\s\S]+)" 
                if self.mode == "baseline" :
                    regex = r"ANSWER:\s*(True|False|Uncertain)"
                    
                if self.model_server.mode == "server":
                    if self.structured=="json": 
                        params["extra_body"] = {
                            "guided_json": json_schema,
                            # "separate_reasoning": True,
                        }
                    elif self.structured=="regex":   
                        params["extra_body"] = {
                            "guided_regex": regex,
                            # "separate_reasoning": True,
                        }
                else:
                    if self.structured=="json": 
                        params["guided_decoding"] = GuidedDecodingParams(
                            json=json_schema
                        )
                    elif self.structured=="regex":  
                        params["guided_decoding"] = GuidedDecodingParams(
                            regex=regex
                        )
                        
            return self.model_server.generate(prompt, params)    
            
        except Exception as e:
            print(f"Error during VLLM API call: {e}")
            return self.ERROR_TOKEN
    
    

    def postprocess_generation(self, gen, completion_only=False):
        """
        Enhanced postprocessing with robust FOL extraction
        """
        try:
            # Extract completion portion
            # gen = generation.strip()
            
            # if self.DEBUG: print(f"Postprocessing Generation for idx {idx}:\n{gen}\n")  # Debugging line

            # Mode-specific processing
            if self.mode in ["neurosymbolic", "neurocot", "neurostep"]:
                return self._process_neurosymbolic(gen)
            elif self.mode in ["cot", "scratchpad",  "baseline"]:
                return self._process_cot(gen)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

        except Exception as e:
            print(f"Postprocessing error: {str(e)}")
            return self.ERROR_TOKEN
    
    
    def extract_fols_neurocot(self, gen: str):
        """
        Strip out any REASONING: lines, then extract exactly one line per FOL: marker,
        and drop anything that doesn’t start with a quantifier or a predicate.
        """
        # 1) drop reasoning lines
        clean = re.sub(r'(?m)^\s*REASONING:.*$', '', gen)

        # 2) grab exactly the text after "FOL:" up to end of that line
        pattern = re.compile(r'^\s*FOL:\s*(.+)$', re.MULTILINE)
        raw = [m.group(1).strip() for m in pattern.finditer(clean)]

        # 3) keep only proper FOL expressions:
        valid = []
        for f in raw:
            # allow:  all x. (...)  OR  exists x. (...)  OR  PredicateName(...)
            if re.match(r'^(all\s+\w+\.|exists\s+\w+\.|[A-Z][A-Za-z0-9_]*\()', f):
                valid.append(f)
        return valid
    
    
    def _process_neurocot(self, gen: str):
        """
        Handles decoding for the 'neurocot' mode, where the model emits
        We drop the REASONING lines, pull out all of the FOLs, then evaluate.
        """
        fols = self.extract_fols_neurocot(gen)
        premises, conclusion = fols[:-1], fols[-1]
        # 4) dispatch to your existing evaluator
        return evaluate(premises, conclusion, self.exp_name)

    def _process_neurosymbolic(self, gen):
     
        
        # print("Gen type is", type(gen))
        # parsed = json.loads(gen)
        
        try:
            parsed = json.loads(gen)
            entries = parsed.get("fol_pairs", [])
            if not entries or not isinstance(entries, list):
                warn("Missing or malformed 'entries' list")
                return self.ERROR_TOKEN

            fols = [e.get("fol", "").strip() for e in entries if "fol" in e]
            if len(fols) < 2:
                warn("Not enough FOL expressions to evaluate")
                return self.ERROR_TOKEN

            premises = fols[:-1]
            conclusion = fols[-1]

            if not all(premises) or not conclusion:
                warn("Empty FOL strings detected")
                return self.ERROR_TOKEN

            return evaluate(premises, conclusion, self.exp_name)

        except (json.JSONDecodeError, ValueError) as e: 
            fol_pattern = re.compile(
                r"^\s*FOL:\s*((?:[^\n]|[\n](?!\s*TEXT:))+?)(?=\s*(?:TEXT:|ANSWER:|$))",
                re.MULTILINE | re.IGNORECASE
            )
            
            matches = fol_pattern.findall(gen)
            
            if matches:
                # Clean and join continued lines
                cleaned_matches = [
                    ' '.join(m.strip().splitlines()) 
                    for m in matches
                ]
                
                # Validate we have at least 1 premise and 1 conclusion
                if len(cleaned_matches) < 2:
                    warn("Insufficient FOL expressions found")
                    return self.ERROR_TOKEN
                    
                premises = cleaned_matches[:-1]
                conclusion = cleaned_matches[-1]
                
                # Validate conclusion syntax
                if any(c in conclusion for c in ['\n', '->>']):
                    warn(f"Malformed conclusion: {conclusion}")
                    return self.ERROR_TOKEN
                    
                # print("Extracted FOL expressions:")
                # print(f"Premises: {premises}")
                # print(f"Conclusion: {conclusion}")    
                # try:
        
                return evaluate(premises, conclusion, self.exp_name)
                # except Exception as e:
                #     print(f"Evaluation error: {str(e)}")
                #     return self.ERROR_TOKEN
            
            warn("No FOL expressions found in generation")
            return self.ERROR_TOKEN

    # def _process_neurosymbolic(self, gen):
    #     """Handles multi-line FOL expressions in conclusions"""
    #     # Improved regex to capture multi-line FOL expressions
       
    def _process_cot(self, gen: str) -> str:
        """
        Handles CoT mode answer extraction, robust to different markers.
        """
        # Try common prefixes first
        prefix_patterns = [
            r"(?:ANSWER|Answer|Final Answer|So the answer is)\s*[:\-]\s*(True|False|Uncertain)",
        ]
        for pat in prefix_patterns:
            m = re.search(pat, gen, flags=re.IGNORECASE)
            if m:
                return m.group(1).capitalize()

        # Fallback: find the last occurrence of one of the labels
        all_labels = re.findall(r"\b(True|False|Uncertain)\b", gen, flags=re.IGNORECASE)
        if all_labels:
            return all_labels[-1].capitalize()

        # nowhere to be found
        return self.ERROR_TOKEN
    
    def verify(self, error_log, prompt):
        if type(error_log) ==str: 
            error_message = error_log
        else:
            error_message = error_log["error_message"]
            premises = error_log["premises"]
            conclusions = error_log["conclusion"]
         
            prompt+= f"Premises: {premises} \n Conclusions: {conclusions}"
            if self.mode in  ["neurosymbolic", "neurocot", "neurostep"]:
                prompt += "However the following errors are generated when passing the FOL statements through Prover9"
                prompt += error_message
                prompt += "Some examples of common errors and fixes: "
                prompt += "Print the correct statements in order to adhere to the Python NLTK package logic module in the same FOL... TEXT... format. "
            
        # else:
        #     prompt += "However there was no "
        #     prompt += error_message
        #     prompt += "Print the correct statements in order to adhere to the Python NLTK package logic module in the same FOL... TEXT... format. "
        
        verified = self.generate_with_vllm(prompt)
        # print(verified)
        return self.postprocess_generation(verified)
        
    @staticmethod
    def metric(generations, references, error_token):
        """Calculates accuracy based on majority voting."""
        correct = 0
        for gens, ref in zip(generations, references):
            gens = [gen for gen in gens if gen != error_token]
            if len(gens) > 0:
                majority = Counter(gens).most_common(1)[0][0]
                if majority == ref:
                    correct += 1
        return {f"accuracy (pass@1 majority)": correct / len(references)}

    def process_results(self, generations, references):
        """
        Processes generations and computes the evaluation metrics.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        :return: dict[str: float]
        """
        return self.metric(generations, references, self.ERROR_TOKEN)
    
    def evaluate(self):
        # print()
        """Incremental evaluation: logs each example as JSONL on the fly."""
        generations = []
        references = []

        exp_dir = os.path.join("results", self.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_path = os.path.join(exp_dir, "progress.json")
        log_path  = os.path.join(exp_dir, "log.jsonl")

        # load already done indices
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as f:
                done = set(json.load(f).keys())
        else:
            done = set()
        # prog = {}
        
        # open JSONL log for appending
        log_file = open(log_path, "a+", encoding="utf-8")
        
        hard_idx = [7, 103, 155, 181, 117, 142, 17, 28, 41, 50, 61, 62, 76, 81, 102, 122, 126, 140, 153, 160, 184, 192, 196, 109, 0, 3, 4, 29, 128, 130]
        hard_id = [0]
        for idx, example in tqdm(enumerate(self.test_dataset),
                                total=len(self.test_dataset),
                                desc="Evaluating"):
            # print(example)
            if str(idx) in done:
                # if str(idx) not in hard_idx:
                continue

            start = time.time()
            
            
            prompt = self.get_prompt(example)

    
            # generate k answers
            example_gens = []
            raw_generations = []
            
            # prompts = [prompt] * self.k
            for i in range(self.k):
                start1 = time.time()
               
                gen = self.generate_with_vllm(prompt)
                # print(gen)
                raw_generations.append(gen)
                end1 = time.time()
                # print("Time for generation: ", end1-start1, "s\n")
                
            elapsed1 = time.time() - start
            
            def _worker(gen):
                # this will run Prover9 under the hood
                gen = self.postprocess_generation(gen)
                if gen in ("True", "False", "Uncertain"):
                    return gen
                
                if self.do_verify==True:
                    i=0
                    while(i<self.NUM_VERIFY):
                        corrected = self.verify(gen, prompt)
                        
                        if corrected in ("True", "False", "Uncertain"):
                            print(f"Verification {i+1} Passed")
                            return corrected
                        else:
                            i+=1
                            continue
                    
                    print("Verification Failed")
                    
                return "Error"
                
            
            with ThreadPoolExecutor(max_workers=5) as pool:
                example_gens = list(pool.map(_worker, raw_generations))


            true_label = example["label"]
            # true_label_is_error = example["error"]
            
            elapsed2 = time.time() - start

            # build entry
            entry = {
                "idx": idx,
                "prompt": prompt,
                "raw_generations": raw_generations, 
                "answers": example_gens,
                "reference": true_label,
                # "error": true_label_is_error,
                "time_total": elapsed2,
                "time_llm": elapsed1,
                "prompt_length": len(prompt)

            }

            # write JSON line and flush
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

            # update in-memory and checkpoint
            done.add(str(idx))
            # we'll rebuild metrics later; for resume we just need keys
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    prog = json.load(f)
            else:
                prog = {}
                
            prog[str(idx)] = {
                "generations": example_gens,
                "reference": true_label
            }
            with open(ckpt_path, "w") as f:
                json.dump(prog, f)

        log_file.close()

        # after all done, load checkpoint for final metrics
        with open(ckpt_path) as f:
            prog = json.load(f)
            
        generations = [v["generations"] for v in prog.values()]
        references  = [v["reference"]   for v in prog.values()]

        # calculate & save results
        metrics = self.process_results(generations, references)
        with open(os.path.join(exp_dir, "results.txt"), "w") as res_file:
            res_file.write(f"Accuracy: {metrics['accuracy']:.2%}\n")
            res_file.write(f"Correct: {metrics['correct']}/{metrics['total']}\n")
            res_file.write(f"Parameters: mode={self.mode}, shots={self.n_shot}, "
                           f"model={self.model_name}, k={self.k}\n")
            res_file.write(f"Notes: {self.notes}")

        return metrics

    @staticmethod
    def metric(generations, references, error_token):
        """Enhanced metric calculation with majority voting"""
        correct = 0
        total = len(references)
        for gens, ref in zip(generations, references):
            valid_gens = [g for g in gens if g != error_token]
            if valid_gens:
                majority = Counter(valid_gens).most_common(1)[0][0]
                correct += (majority == ref)
        # if Counter(gens).most_common(1)[0][0] == error_token and 
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total
        }

    def process_results(self, generations, references):
        """Process results using new metric"""
        return self.metric(generations, references, self.ERROR_TOKEN)
