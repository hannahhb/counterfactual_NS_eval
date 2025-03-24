from functools import cache
from collections import Counter
from eval.tasks.utils import evaluate, convert_to_nltk_rep
from abc import abstractmethod, ABC
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
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

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

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
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
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


class FOLIOTask(Task):
    """A First-Order Logic Inference Task following the Open World Assumption."""

    DATASET_PATH = "yale-nlp/FOLIO"
    DATASET_NAME = "train"
    MODES = ["baseline", "cot", "scratchpad", "neurosymbolic"]
    ERROR_TOKEN = "Error"
    MAX_SHOTS = 16
    qwen = "Qwen/Qwen2.5-7B-Instruct"
    mistral = "mistralai/Mistral-7B-Instruct-v0.3"
    starcoder = "bigcode/starcoder2-7b"
    DEFAULT_MODEL_NAME = starcoder

    def __init__(self, mode="cot", n_shot=3, model_name=DEFAULT_MODEL_NAME):
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

        super().__init__(
            stop_words=["</EVALUATE>"], requires_execution=(mode == "neurosymbolic"),
        )

        self.mode = mode
        self.n_shot = n_shot
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model()
        self.train_dataset = self.prepare_train_dataset()
        self.test_dataset = load_dataset(self.DATASET_PATH, split='train').select(range(3))  # Adjust as needed

    def load_model(self):
        """Loads the language model and tokenizer."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def prepare_train_dataset(self):
        """Prepares the training dataset with necessary preprocessing."""
        train_dataset = load_dataset(self.DATASET_PATH, split='train')

        # Reformat premises to NLTK representations
        train_dataset = train_dataset.map(self.reformat_fol_samples_train)

        # Add conclusion FOL expressions
        train_dataset = self.add_conclusion_fols_train(train_dataset)

        # Add Chain-of-Thought (CoT) explanations
        train_dataset = self.add_cot_train(train_dataset)

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
        """Converts premises to NLTK logical representations."""
        sample["premises-FOL"] = [
            convert_to_nltk_rep(premise) for premise in sample["premises"]
        ]
        return sample

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
            683: "exists x. (BRICS(x) & Speaks(x, Hindi))",
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
        # if "conclusion-FOL" in train_dataset.column_names:
        #     train_dataset = train_dataset.remove_columns("conclusion-FOL")
        
        # # Add the new column
        # train_dataset = train_dataset.add_column("conclusion-FOL", conclusions)
                 
        train_dataset = train_dataset.add_column("cot", cots)
        return train_dataset

    @cache
    def fewshot_examples(self):
        """Generates few-shot examples for the prompt."""
        examples = []
        for doc in self.train_dataset:
            examples.append(self.format_train_example(doc))
        return "\n".join(examples)

    def format_train_example(self, doc):
        """Formats a single training example based on the mode."""
        example = self.format_test_example(doc)
        if self.mode == "baseline":
            example += f"{doc['label'].strip()}\n"
        elif self.mode == "cot":
            example += f"{doc['cot']}\n"
        else:
            for premise, fol in zip(doc["premises"], doc["premises-FOL"]):
                example += f"TEXT:\t{premise.strip()}\nFOL:\t{fol.strip()}\n"
            example += f"TEXT:\t{doc['conclusion'].strip()}\nFOL:\t{doc['conclusion-FOL'].strip()}\n"
            if self.mode == "scratchpad":
                example += f"ANSWER:\t{doc['label'].strip()}\n"
        return example + "</EVALUATE>\n"

    def format_test_example(self, doc):
        """Formats a single test example."""
        example = "<PREMISES>\n"
        for premise in doc["premises"]:
            example += f"{premise.strip()}\n"
        example += "</PREMISES>\n"
        example += f"<CONCLUSION>\n{doc['conclusion'].strip()}\n</CONCLUSION>\n"
        example += "<EVALUATE>\n"
        return example

    def get_dataset(self):
        """Returns the test dataset."""
        return self.test_dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        instructions = self.get_instructions()
        train = self.fewshot_examples()
        test = self.format_test_example(doc)
        prompt = "\n".join([instructions, train, test])
        # print(prompt)
        return prompt

    def get_instructions(self):
        """Generates task-specific instructions based on the mode."""
        instructions = ""
        instructions += "The following is a first-order logic (FOL) problem.\n"
        instructions += "The problem is to determine whether the conclusion follows from the premises.\n"
        instructions += "The premises are given in the form of a set of first-order logic sentences.\n"
        instructions += "The conclusion is given in the form of a single first-order logic sentence.\n"
        if self.mode == "baseline":
            instructions += "The task is to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
        else:
            instructions += "The task is to translate each of the premises and conclusions into FOL expressions, "
            if self.mode == "scratchpad":
                instructions += "and then to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
            elif self.mode == "neurosymbolic":
                instructions += "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.\n"
                instructions += "Expressions should adhere to the format of the Python NLTK package logic module."
            # elif self.mode == "cot":
            #     instructions += "Provide your reasoning step by step and conclude with 'ANSWER:\tTrue', 'ANSWER:\tFalse', or 'ANSWER:\tUncertain'."
        return instructions + "\n\n"

    def get_reference(self, doc):
        """
        Retrieves the ground truth label for a given document.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["label"]

    def postprocess_generation(self, generation, idx, completion_only=False):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        try:
            if completion_only:
                gen = generation.strip()
            else:
                prefix = self.get_prompt(self.get_dataset()[idx])
                assert generation.startswith(
                    prefix
                ), "Increase `--max_length_generation` to avoid truncation"
                gen = generation[len(prefix):].strip()
                for stop_word in self.stop_words:
                    gen = gen.split(stop_word)[0].strip()

            if self.mode == "baseline":
                resp = gen.strip()
            elif self.mode == "scratchpad":
                flag = "ANSWER:"
                resp = gen.split(flag)[-1].strip()
            elif self.mode == "neurosymbolic":
                flag = "FOL:"
                parses = [
                    line.replace(flag, "").strip()
                    for line in gen.split("\n")
                    if flag in line
                ]
                premises, conclusion = parses[:-1], parses[-1]
                resp = evaluate(premises, conclusion)
            elif self.mode == "cot":
                flag = "ANSWER:"
                resp = gen.split(flag)[-1].strip()
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            assert resp in ["True", "False", "Uncertain"], f"Invalid generation: {resp}"
            return resp
        except Exception as e:
            print(f"Error in parsing and/or evaluating LLM output: {e}")
            return self.ERROR_TOKEN

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
        """Evaluates the model on the test dataset."""
        results = {"correct": 0, "total": 0}
        generations = []
        references = []

        for idx, example in enumerate(self.test_dataset):
            prompt = self.get_prompt(example)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            output = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                # top_p=0.9,              # Added for better sampling
                # top_k=50,               # Added for better sampling
                # repetition_penalty=1.2, # Added to prevent repetition
                pad_token_id=self.tokenizer.eos_token_id
            )

            generation = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self.postprocess_generation(generation, idx)
            print(f"Generated Answer: {answer}")

            true_label = self.get_reference(example)
            generations.append([answer])
            references.append(true_label)

            results["total"] += 1
            if answer.lower() == true_label.lower():
                results["correct"] += 1

        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
        return results


# Usage Example
if __name__ == "__main__":
    task = FOLIOTask(mode="cot", n_shot=3)
    evaluation_results = task.evaluate()
    print(f"Evaluation Results: {evaluation_results}")