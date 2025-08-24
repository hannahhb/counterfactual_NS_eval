from abc import abstractmethod, ABC
from datasets import load_dataset
from warnings import warn

class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            whether the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        except:
            warn(
                "This task will use a locally downloaded dataset, not from the HF hub."
            )

    

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass


    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        :return: dict[str: float]
        """
        pass

# class OWAFOLTask(Task):
#     """A First-Order Logic Inference Task following the Open World Assumption."""

#     # DATASET_PATH = "minimario/FOLIO"
#     DATASET_PATH = "yale-nlp/FOLIO"
#     DATASET_NAME = "train"
#     MODES = ["baseline", "cot", "scratchpad", "neurosymbolic"]
#     ERROR_TOKEN = "Error"
#     MAX_SHOTS = 16
#     DEBUG = False

    
#     port = 8000
    
#     def __init__(self, model_name, model_server,  mode="baseline", n_shot=3, k=5, run=1, dataset_type="counterfactual"):
#         """
#         :param mode: str
#             Inference mode. One of ["baseline", "cot", "scratchpad", "neurosymbolic"]
#         :param n_shot: int
#             Number of few-shot examples to use
#         :param model_name: str
#             HuggingFace model name
#         """
#         assert mode in self.MODES, f"Invalid mode. Choose from {self.MODES}"
#         assert n_shot <= self.MAX_SHOTS, f"Supports up to {self.MAX_SHOTS}-shot"

#         super().__init__(
#             stop_words=["</EVALUATE>"], requires_execution=(mode == "neurosymbolic"),
#         )

#         # self.test_data_name = test_data
#         self.base_exp_name = f"{model_name}/{mode}_k{k}"
#         # counter_path = "data/folio_v2_perturbed.jsonl"
            
#         self.mode = mode
#         self.n_shot = n_shot
#         self.model_name = model_name
#         self.k = k
        
#         self.dataset_type = dataset_type
#         self.train_dataset = self.prepare_train_dataset()
#         self.exp_name, self.run = self.get_unique_exp_name(self.base_exp_name, run, dataset_type)
#         self.model_server = model_server


#     def prepare_train_dataset(self):
#         """Prepares the training dataset with necessary preprocessing."""
#         train_dataset = load_dataset(self.DATASET_PATH, split='train')

#         # Reformat premises to NLTK representations
#         train_dataset = train_dataset.map(self.reformat_fol_samples_train)

#         # Add conclusion FOL expressions
#         train_dataset = self.add_conclusion_fols_train(train_dataset)

#         # Add Chain-of-Thought (CoT) explanations
#         train_dataset = self.add_cot_train(train_dataset)

#         # Map labels to standardized format
#         train_dataset = train_dataset.map(
#             lambda x: {"label": "Uncertain" if x["label"] == "Unknown" else x["label"]},
#             remove_columns=["label"],
#         )

#         # Select few-shot examples based on predefined indices
#         fewshot_indices_all = [
#             125, 23, 60, 275, 148, 261, 263, 683, 299, 684, 850, 853, 886, 892, 930, 980,
#         ]
#         fewshot_indices = fewshot_indices_all[:self.n_shot]
#         train_fewshot = train_dataset.select(fewshot_indices)
#         return train_fewshot

#     def reformat_fol_samples_train(self, sample):
#         """Properly handles multi-line premises"""
#         # Split premises into individual lines and clean
#         premises = [p.strip() for p in sample["premises-FOL"].split("\n") if p.strip()]
        
#         # Convert each premise to FOL
#         sample["premises-FOL"] = [
#             convert_to_nltk_rep(premise) 
#             for premise in premises
#         ]
#         return sample

    
#     def format_test_example(self, doc, is_default = True):
#         """Fixed test example formatting"""
#         example = "<PREMISES>\n"
#         # Properly split multi-line premises
#         prem = "premises"
#         concl = 'conclusion'
        
#         if self.dataset_type == "default" and is_default:  
#             prem = "orig_" + prem
#             concl = "orig_" + concl
#         # print(doc)
#         # print(doc[prem])
        
#         premises = [p.strip() for p in doc[prem].split("\n") if p.strip()]
#         # print(premises)
#         for premise in premises:
#             example += f"{premise}\n"  
      
#         example += "</PREMISES>\n"
#         example += f"<CONCLUSION>\n{doc[concl].strip()}\n</CONCLUSION>\n"
#         example += "<EVALUATE>\n"
#         return example
    
#     def format_train_example(self, doc):
#         """Fixed formatting for training examples"""
#         example = self.format_test_example(doc, is_default=False)
        
#         # Handle premises properly
#         if self.mode == "baseline":
#             example += f"ANSWER: {doc['label'].strip()}\n"
#         elif self.mode == "cot":
#             example += f"\n{doc['cot']}\n"
#         else:
#             premises = [p.strip() for p in doc["premises"].split("\n") if p.strip()]

#             for premise, fol in zip(premises, doc["premises-FOL"]):
#                 example += f"TEXT: {premise}\nFOL: {fol}\n"
        
#             example += f"TEXT: {doc['conclusion'].strip()}\n"
#             example += f"FOL: {doc['conclusion-FOL'].strip()}\n"
#             if self.mode == "scratchpad":
#                 example += f"\nANSWER: {doc['label'].strip()}\n"
        
#         return example + "</EVALUATE>\n"
    
#     def add_conclusion_fols_train(self, train_dataset):
#         """Adds conclusion FOL expressions to the training dataset."""
#         train_conclusion_fols = {
#             23: "HigherRank(RealMadrid, Barcelona)",
#             60: "-OlympicGoldMedalWinner(Amy) -> NobelLaureate(Amy)",
#             125: "-Dispensable(Worksheet)",
#             148: "FolkSong(Inception)",
#             261: "MakeGoodBreakfast(Luke)",
#             263: "exists x. (Develops(Ets, x) & For(x, k-OneTwoandhighereducation)) & exists x. (Develops(Ets, x) & AssociatedWith(x, Entrytouseducationinstitutions))",
#             275: "ContributeToCountry(James)",
#             299: "GetRhythmRight(John)",
#             683: "exists x. (BRICS(x) & Speaks(x, Hindi))",
#             684: "Film(Hamilton)",
#             850: "-Liked(Leo, Charlie) & -Cares(Charlie, Leo)",
#             853: "Won(Threebodyproblem, Hugoaward)",
#             886: "Dagfinn(DagfinnAarskog)",
#             892: "PartOf(Minsk, Scottishpremiership)",
#             930: "-Locate(Boves, Europe)",
#             980: "(InvitedTakePhoto(James) & -HappyCommunicate(James)) | (-InvitedTakePhoto(James) & HappyCommunicate(James))",
#         }
#         conclusions = [None for _ in range(len(train_dataset))]
#         for index, conclusion_fol in train_conclusion_fols.items():
#             if index < len(conclusions):
#                 conclusions[index] = conclusion_fol

#         # Remove the column if it exists
#         if "conclusion-FOL" in train_dataset.column_names:
#             train_dataset = train_dataset.remove_columns("conclusion-FOL")
        
#         # Add the new column
#         train_dataset = train_dataset.add_column("conclusion-FOL", conclusions)
#         return train_dataset
    
#     def add_cot_train(self, train_dataset):
#         """Adds Chain-of-Thought explanations to the training dataset."""
#         train_cots = {
#             23: "Let's think step by step. We want to evaluate if in La Liga 2021-2022, Real Madrid ranks higher than Barcelona. From premise 1, we know that a La Liga soccer team ranks higher than another if it receives more points. From premise 4, we know that in La Liga 2021-2022, Real Madrid received more points than Barcelona. Therefore, in La Liga 2021-2022, Real Madrid received more points than Barcelona, so Real Madrid ranks higher than Barcelona, so the statement is true.\nANSWER:\tTrue",
#             60: "Let's think step by step. We want to evaluate the statement \"if Amy is not an Olympic gold medal winner, then Amy is a Nobel laureate\". Let's assume that Amy is not an Olympic gold medal winner. This doesn't tell us anything about whether Amy is a Nobel laureate, so the statement isn't true, meaning it is either False or Uncertain. To distinguish between the two, notice that we could have a scenario where Amy is neither an Olympic gold medal winner nor a Nobel laureate. None of the premises are violated in this case. This means the statement must be false.\nANSWER:\tFalse",
#             125: "Let's think step by step. We want to evaluate if a worksheet is not dispensable. From premise 6, we know that a worksheet is either paper or is environment-friendly. If it is paper, then from premise 3, a worksheet is woodware, and from premise 2, a worksheet is dispensable. If it is environment-friendly, we know it is good from premise 5, but we know nothing about whether it is dispensable. Therefore, we don't know if a worksheet is dispensible or not, so the statement is uncertain.\nANSWER:\tUncertain",
#             148: "Let's think step by step. We want to evaluate if Inception is a folk song. We know that Inception is a sci-fi movie. Since all movies are videos and Inception is a movie, it is a video, which means it is visual. On the other hand, we know that all folk songs are songs, and no songs are visual, so no folk songs are visual. Therefore, since Inception is visual but no folk songs are visual, we know that Inception cannot be a folk song, so the statement is false.\nANSWER:\tFalse",
#             261: "Let's think step by step. We want to evaluate if Luke can make a good breakfast. From the last premise, we know that Luke can make cookies, scrambled eggs, and muffins. Since Luke can make cookies and muffins, they are a baker. Now, combining the information we have, since Luke is a baker and can make scrambled eggs, this means that they can make a good breakfast. Therefore, Luke can make a good breakfast, so the statement is true.\nANSWER:\tTrue",
#             263: "Let's think step by step. We want to evaluate if ETS develops assessments for K-12 statewide as well as entry to US tertiary and quaternary educatiand doon institutions. We know that ETS develops assessments for K-12 statewide. We also know that ETS develops assessments associated with entry to the US tertiary and quaternary education institutes. Therefore, both parts of the conclusion are true, and the statement is true.\nANSWER:\tTrue",
#             275: "Let's think step by step. We want to evaluate if James contributes to the country. Let's think about what we know about James. First, we know that James was either sentenced for thief or stayed in prison. However, this doesn't tell us anything about whether James contributed to the country. Second, we know that James either had a bad record in the local state or that he was respected by others. However, the premises don't tell us anything about the relationship between having a bad record and contributing to the country. Therefore, it is uncertain whether James contributes to the country.\nANSWER:\tUncertain",
#             299: "Let's think step by step. We want to evaluate if John can get the rhythms right. We know that John is a student learning piano. Since all students learning piano can strike the right notes, John can strike the right notes. Since all students who can strike the right notes can get the rhythms right and John can strike the right notes, John can get the rhythms right, so the conclusion is true.\nANSWER:\tTrue",
#             683: "Let's think step by step. We want to evaluate if there is a person from BRICS speaking Hindi. We know that there is an Indian, and since India is one of BRICS, we know that there is an Indian in BRICS. Furthermore, we know that they speak either Hindi or English, however, we don't know which one. Therefore, there could be a person in BRICS speaking Hindi, or there could not. Therefore, it is uncertain whether there is a person from BRICS speaking Hindi.\nANSWER:\tUncertain",
#             684: "Let's think step by step. We want to evaluate if Hamilton is a film. Since Daveed Diggs played two roles in the musical Hamilton, Hamilton is a musical. Since musicals are not films and Hamilton is a musical, Hamilton is not a film, and the conclusion is false.\nANSWER:\tFalse",
#             850: "Let's think step by step. We want to evaluate if Charlie does not like Leo and does not care for Leo. Let's first evaluate if Charlie does not like Leo. We know Charlie has a naughty pet named Leo. Since pets who are naughty are not liked as much, Charlie does not like Leo. Now, let's evaluate if Charlie cares for Leo. We know that if a person has a pet, they care for that pet. Since Leo is Charlie's pet, Charlie cares for Leo. Therefore, Charlie does not like Leo but cares for Leo, so the second part of the conclusion is false, which means the entire conclusion is false.\nANSWER:\tFalse",
#             853: "Let's think step by step. We want to evaluate if the Three Body Problem won the Hugo Award. The only thing we know about the Hugo Award is that some books that have won the Hugo Award were written by Cixin Liu. However, we know nothing about whether The Three Body Problem was written by Cixin Liu, so the conclusion is uncertain.\nANSWER:\tUncertain",
#             886: "Let's think step by step. We want to evaluate if Dagfinn is Dagfinn Aarskog's given name. We know that Dagfinn is a given name, and that notable people with the given name Dagfinn includes Dagfinn Aarskog, which means that Dagfinn is Dagfinn Aarskog's given name, so the conclusion is true.\nANSWER:\tTrue",
#             892: "Let's think step by step. We want to evaluate if Minsk joined the Scottish Premiership. We know that Minsk and St Johnstone are different teams and that St Johnstone is part of the Scottish Premiership, but we don't know anything about whether or not Minsk joined the Scottish Premiership from the premises. Therefore, the conclusion is uncertain.\nANSWER:\tUncertain",
#             930: "Let's think step by step. We want to evaluate if Boves is not in Europe. We know that Boves is a railway station located in France. We also know that since France is a European country, France is located in Europe. Furthermore, we know that if A is located in B and B is located in C, then A is located in C. Therefore, we know that because Boves is located in France and France is located in Europe, that means Boves is located in Europe. Therefore, the conclusion is false.\nANSWER:\tFalse",
#             980: "Let's think step by step. We want to evaluate if James is either invited to take a photo with the audience or happy to communicate with each other during the dinner. We know that James does not attend the conference in person and is not provided with souvenirs. There are no premises that apply to people who do not attend the conference. Since James is not provided with souvenirs, since all who attended the conference in person are provided with souvenirs, we know that James did not attend the conference in person. However, we don't know anything else, so it is possible that James was neither invited to take a photo with the audience nor happy to communicate during the dinner. Therefore, the conclusion is false.\nANSWER:\tFalse",
#         }
#         cots = [None for _ in range(len(train_dataset))]
#         for index, cot in train_cots.items():
#             if index < len(cots):
#                 cots[index] = cot
                
#         train_dataset = train_dataset.add_column("cot", cots)
#         return train_dataset

#     @cache
#     def fewshot_examples(self):
#         """Generates few-shot examples for the prompt."""
#         examples = []
#         for doc in self.train_dataset:
#             examples.append(self.format_train_example(doc))
#         return "\n".join(examples)

    

#     def get_prompt(self, doc):
#         """
#         Builds the prompt for the LM to generate from.
#         :param doc: dict[str: str]
#             sample from the test dataset
#         :return: str
#         """
#         instructions = self.get_instructions()
#         train = self.fewshot_examples()
#         test = self.format_test_example(doc)
#         prompt = "\n".join([instructions, train, test])
#         return prompt

#     def get_instructions(self):
#         """Generates task-specific instructions based on the mode."""
#         instructions = ""
#         instructions += "The following is a first-order logic (FOL) problem.\n"
#         instructions += "The problem is to determine whether the conclusion follows from the premises.\n"
#         instructions += "The premises are given in the form of a set of first-order logic sentences.\n"
#         instructions += "The conclusion is given in the form of a single first-order logic sentence.\n"
#         if self.mode == "baseline":
#             instructions += "The task is to evaluate the conclusion as ANSWER: True, ANSWER: False, or ANSWER: Uncertain given the premises. No need for extra output."
#         else:
#             instructions += "The task is to translate each of the premises and conclusions into FOL expressions "
#             if self.mode == "scratchpad":
#                 instructions += "and then to evaluate the conclusion as ANSWER: True, ANSWER: False, or ANSWER: Uncertain given the premises." 
#                 # "The output should follow format TEXT: \nFOL: \nTEXT: \nFOL:... ANSWER: "
#             elif self.mode == "neurosymbolic":
#                 # if self.model_name == self.deepseek:
#                 #     instructions += "\n\nLimit thinking to 512 chars. Provide the premises and conclusion in the following format:\n" \
#                 #     "TEXT: <premise text>\nFOL: <first-order logic expression>\n" \
#                 #     "TEXT: <conclusion text>\nFOL: <first-order logic expression>"
#                 # else:
#                     instructions += "so that the expressions can be evaluated by a theorem solver like Prover9 to determine whether the conclusion follows from the premises.\n"
#                     instructions += "Expressions should adhere to the format of the Python NLTK package logic module. "
#             # instructions += "The output should follow format TEXT: \nFOL: \nTEXT: \nFOL:..." # new addition to ensure format
#                     # if self.dataset_type=="counterfactual": 
                        
#             # elif self.mode == "cot":
#             #     instructions += "Provide your reasoning step by step and conclude with 'ANSWER:\tTrue', 'ANSWER:\tFalse', or 'ANSWER:\tUncertain'."
#         return instructions + "\n\n"
 
    
        
#     def get_unique_exp_name(self, base_path, run=1, suffix=""):
#         """Find a unique experiment name by incrementing run number."""
#         while True:
#             exp_name = f"{base_path}_run{run}_{suffix}"
#             exp_dir = Path("results") / exp_name
            
#             if not exp_dir.exists():
#                 return exp_name, run
#             run += 1
        
#     def generate_with_vllm(self, prompt):
#         """Generates text from the vLLM using the OpenAI client."""
        
#         try:
#             params = {
#                 "temperature": 0.8,
#                 # "max_tokens": 4096
#                 "top_p": 0.92, 
#                 "stop": self.stop_words
 
#             }
            
#             if self.mode == "neurosymbolic":
#                 # regex =  r"TEXT:\s*(.+?)\s*FOL:\s*((?:[^\n]|[\n](?!\s*TEXT:))+)"
#                 regex = r"TEXT:\s*(.+?)\s*FOL:\s*([\s\S]+)"  

#                 if self.model_server.mode == "server":
#                     params["extra_body"] = {
#                         "regex": regex,
#                         "separate_reasoning": True,
#                     }
#                 else:
#                     params["guided_decoding"] = GuidedDecodingParams(
#                         regex=regex
#                     )
                    
#             if self.mode == "baseline":
#                 regex = r"ANSWER:\s*(True|False|Uncertain)"
#                 if self.model_server.mode == "server":
#                     params["extra_body"] = {"regex": regex}
#                 else:
#                     params["guided_decoding"] = GuidedDecodingParams(
#                         regex=regex
#                     )
#             return self.model_server.generate(prompt, params)    
            
#         except Exception as e:
#             print(f"Error during VLLM API call: {e}")
#             return self.ERROR_TOKEN

#     def postprocess_generation(self, gen, idx, completion_only=False):
#         """
#         Enhanced postprocessing with robust FOL extraction
#         """
#         try:
#             # Extract completion portion
#             # gen = generation.strip()
            
#             if self.DEBUG: print(f"Postprocessing Generation for idx {idx}:\n{gen}\n")  # Debugging line

#             # Mode-specific processing
#             if self.mode == "neurosymbolic":
#                 return self._process_neurosymbolic(gen)
#             elif self.mode == "cot" or "scratchpad" or "baseline":
#                 return self._process_cot(gen)
#             else:
#                 raise ValueError(f"Invalid mode: {self.mode}")

#         except Exception as e:
#             print(f"Postprocessing error: {str(e)}")
#             return self.ERROR_TOKEN

#     def _process_neurosymbolic(self, gen):
#         """Handles multi-line FOL expressions in conclusions"""
#         # Improved regex to capture multi-line FOL expressions
#         fol_pattern = re.compile(
#             r"^\s*FOL:\s*((?:[^\n]|[\n](?!\s*TEXT:))+?)(?=\s*(?:TEXT:|ANSWER:|$))",
#             re.MULTILINE | re.IGNORECASE
#         )
        
#         matches = fol_pattern.findall(gen)
        
#         if matches:
#             # Clean and join continued lines
#             cleaned_matches = [
#                 ' '.join(m.strip().splitlines()) 
#                 for m in matches
#             ]
            
#             # Validate we have at least 1 premise and 1 conclusion
#             if len(cleaned_matches) < 2:
#                 warn("Insufficient FOL expressions found")
#                 return self.ERROR_TOKEN
                
#             premises = cleaned_matches[:-1]
#             conclusion = cleaned_matches[-1]
            
#             # Validate conclusion syntax
#             if any(c in conclusion for c in ['\n', '->>']):
#                 warn(f"Malformed conclusion: {conclusion}")
#                 return self.ERROR_TOKEN
                
#             # print("Extracted FOL expressions:")
#             # print(f"Premises: {premises}")
#             # print(f"Conclusion: {conclusion}")    
#             try:
#                 return evaluate(premises, conclusion, self.exp_name)
#             except Exception as e:
#                 print(f"Evaluation error: {str(e)}")
#                 return self.ERROR_TOKEN
        
#         warn("No FOL expressions found in generation")
#         return self.ERROR_TOKEN
    
#     def _process_cot(self, gen):
#         """Handles CoT mode answer extraction"""
#         # Look for final answer marker
#         answer_match = re.search(
#             r"ANSWER:\s*((True|False|Uncertain))", 
#             gen, 
#             re.IGNORECASE
#         )
#         # print(answer_match)
#         if answer_match:
#             return answer_match.group(1).capitalize()
#         return self.ERROR_TOKEN
    
#     @staticmethod
#     def metric(generations, references, error_token):
#         """Calculates accuracy based on majority voting."""
#         correct = 0
#         for gens, ref in zip(generations, references):
#             gens = [gen for gen in gens if gen != error_token]
#             if len(gens) > 0:
#                 majority = Counter(gens).most_common(1)[0][0]
#                 if majority == ref:
#                     correct += 1
#         return {f"accuracy (pass@1 majority)": correct / len(references)}

#     def process_results(self, generations, references):
#         """
#         Processes generations and computes the evaluation metrics.
#         :param generations: list(list(str))
#             list of lists containing generations
#         :param references: list(str)
#             list of str containing references
#         :return: dict[str: float]
#         """
#         return self.metric(generations, references, self.ERROR_TOKEN)
    
#     def evaluate(self):
#         """Incremental evaluation: logs each example as JSONL on the fly."""
#         generations = []
#         references = []

#         exp_dir = os.path.join("results", self.exp_name)
#         os.makedirs(exp_dir, exist_ok=True)
#         ckpt_path = os.path.join(exp_dir, "progress.json")
#         log_path  = os.path.join(exp_dir, "log.jsonl")

#         # load already done indices
#         if os.path.exists(ckpt_path):
#             with open(ckpt_path, "r") as f:
#                 done = set(json.load(f).keys())
#         else:
#             done = set()

#         # open JSONL log for appending
#         log_file = open(log_path, "a+", encoding="utf-8")

#         for idx, example in tqdm(enumerate(self.test_dataset),
#                                 total=len(self.test_dataset),
#                                 desc="Evaluating"):
#             if str(idx) in done:
#                 continue

#             start = time.time()
#             prompt = self.get_prompt(example)

#             # generate k answers
#             example_gens = []
#             raw_generations = []
            
#             for i in range(self.k):
#                 gen = self.generate_with_vllm(prompt)
#                 ans = self.postprocess_generation(gen, idx)
#                 example_gens.append(ans)
#                 raw_generations.append(gen)

#             true_label = example["label"]
#             elapsed = time.time() - start

#             # build entry
#             entry = {
#                 "idx": idx,
#                 "prompt": prompt,
#                 "raw_generations": raw_generations, 
#                 "answers": example_gens,
#                 "reference": true_label,
#                 "time_s": elapsed
#             }

#             # write JSON line and flush
#             log_file.write(json.dumps(entry) + "\n")
#             log_file.flush()

#             # update in-memory and checkpoint
#             done.add(str(idx))
#             # we'll rebuild metrics later; for resume we just need keys
#             if os.path.exists(ckpt_path):
#                 with open(ckpt_path, "r") as f:
#                     prog = json.load(f)
#             else:
#                 prog = {}
#             prog[str(idx)] = {
#                 "generations": example_gens,
#                 "reference": true_label
#             }
#             with open(ckpt_path, "w") as f:
#                 json.dump(prog, f)

#         log_file.close()

#         # after all done, load checkpoint for final metrics
#         with open(ckpt_path) as f:
#             prog = json.load(f)
#         generations = [v["generations"] for v in prog.values()]
#         references  = [v["reference"]   for v in prog.values()]

#         # calculate & save results
#         metrics = self.process_results(generations, references)
#         with open(os.path.join(exp_dir, "results.txt"), "w") as res_file:
#             res_file.write(f"Accuracy: {metrics['accuracy']:.2%}\n")
#             res_file.write(f"Correct: {metrics['correct']}/{metrics['total']}\n")
#             res_file.write(f"Parameters: mode={self.mode}, shots={self.n_shot}, "
#                            f"model={self.model_name}, k={self.k}\n")

#         return metrics

#     @staticmethod
#     def metric(generations, references, error_token):
#         """Enhanced metric calculation with majority voting"""
#         correct = 0
#         total = len(references)
#         for gens, ref in zip(generations, references):
#             valid_gens = [g for g in gens if g != error_token]
#             if valid_gens:
#                 majority = Counter(valid_gens).most_common(1)[0][0]
#                 correct += (majority == ref)
#         return {
#             "accuracy": correct / total if total > 0 else 0,
#             "correct": correct,
#             "total": total
#         }

#     def process_results(self, generations, references):
#         """Process results using new metric"""
#         return self.metric(generations, references, self.ERROR_TOKEN)


# class FOLIOBase(OWAFOLTask):
#     def __init__(self, model_name, model_server, mode="baseline",
#                  n_shot=3, k=5, run=1, dataset_type="counterfactual"):
#         super().__init__(model_name, model_server, mode, n_shot, k, run,
#                          dataset_type=dataset_type)
#         counter_path = "data/folio_v2_perturbed.jsonl"
#         self.test_dataset = self.load_jsonl_dataset(counter_path)
#         self.exp_name += "_folio"
    
    
#     def load_jsonl_dataset(self, file_path):
#         """
#         Load a JSONL file directly without using the datasets library.
        
#         Args:
#             file_path: Path to the JSONL file
            
#         Returns:
#             List of dictionaries containing the dataset examples
#         """
#         dataset = []
#         file_path = Path(file_path)
        
#         if not file_path.exists():
#             print(f"Warning: File {file_path} not found")
#             return []
            
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     if line.strip():  # Skip empty lines
#                         example = json.loads(line)
#                         dataset.append(example)
            
#             print(f"Successfully loaded {len(dataset)} examples from {file_path}")
#             return dataset
#         except Exception as e:
#             print(f"Error loading JSONL file: {e}")
#             return []
        
# class ProofWriterTask(OWAFOLTask):
#     """Use ProofWriter’s balanced deduction test split as the test set."""
#     PROOFWRITER_HF = "theoxo/proofwriter-deduction-balanced"

#     def __init__(self, model_name, model_server, mode="baseline",
#                  n_shot=3, k=5, run=1, seed=7):
#         super().__init__(model_name, model_server, mode, n_shot, k, run,
#                          dataset_type="default")
#         pw = load_dataset(self.PROOFWRITER_HF, split="test").select(range(100))
#         # apply your reformat_proofwriter helper
#         self.test_dataset = self.reformat_proofwriter(pw).shuffle(seed)
#         self.exp_name += "_proofwriter"      # optional suffix

#     def reformat_proofwriter(self, dataset):

#         def punctuate(s):
#             if s[-1] not in [".", "?", "!"]:
#                 s += "."
#             return s

#         def reformat_sample(sample):
#             sample["premises"] = [punctuate(p) for p in sample.pop("theory").split(". ")]
#             sample["conclusion"] = punctuate(sample.pop("question"))
#             sample["label"] = sample.pop("answer")
#             return sample

#         return dataset.map(reformat_sample)
    
#     def format_test_example(self, doc, is_default = True):
#         """Fixed test example formatting"""
#         example = "<PREMISES>\n"
#         # Properly split multi-line premises
#         prem = "premises"
#         concl = 'conclusion'
        
#         premises = doc[prem]
#         # print(premises)
        
#         if isinstance(premises, list):
#             # already a list of sentences
#             premises = premises
#         else:
#             # split a multiline string into lines, drop blank lines
#             premises = [line.strip() for line in premises.splitlines() if line.strip()]

#         for premise in premises:
#             example += f"{premise}"  
#             example += "\n"
      
#         example += "</PREMISES>\n"
#         example += f"<CONCLUSION>\n{doc[concl].strip()}\n</CONCLUSION>\n"
#         example += "<EVALUATE>\n"
#         return example