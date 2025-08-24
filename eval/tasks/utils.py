import re
import nltk
from nltk.sem import logic
from nltk.sem import Expression
import datetime
import os
import json 
import traceback

logic._counter._value = 0
read_expr = Expression.fromstring
prover = nltk.Prover9(10)


def convert_to_nltk_rep(logic_formula):
    translation_map = {
        "∀": "all ",
        "∃": "exists ",
        "→": "->",
        "¬": "-",
        "∧": "&",
        "∨": "|",
        "⟷": "<->",
        "↔": "<->",
        "0": "Zero",
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        ".": "Dot",
        "Ś": "S",
        "ą": "a",
        "’": "",
        "! ": "", 
        "!": "",
    }

    constant_pattern = r'\b([a-z]{2,})(?!\()'
    logic_formula = re.sub(constant_pattern, lambda match: match.group(1).capitalize(), logic_formula)

    for key, value in translation_map.items():
        logic_formula = logic_formula.replace(key, value)

    quant_pattern = r"(all\s|exists\s)([a-z])"
    def replace_quant(match):
        return match.group(1) + match.group(2) + "."
    logic_formula = re.sub(quant_pattern, replace_quant, logic_formula)

    dotted_param_pattern = r"([a-z])\.(?=[a-z])"
    def replace_dotted_param(match):
        return match.group(1)
    logic_formula = re.sub(dotted_param_pattern, replace_dotted_param, logic_formula)

    simple_xor_pattern = r"(\w+\([^()]*\)) ⊕ (\w+\([^()]*\))"
    def replace_simple_xor(match):
        return ("((" + match.group(1) + " & -" + match.group(2) + ") | (-" + match.group(1) + " & " + match.group(2) + "))")
    logic_formula = re.sub(simple_xor_pattern, replace_simple_xor, logic_formula)

    complex_xor_pattern = r"\((.*?)\)\) ⊕ \((.*?)\)\)"
    def replace_complex_xor(match):
        return ("(((" + match.group(1) + ")) & -(" + match.group(2) + "))) | (-(" + match.group(1) + ")) & (" + match.group(2) + "))))")
    logic_formula = re.sub(complex_xor_pattern, replace_complex_xor, logic_formula)

    special_xor_pattern = r"\(\(\((.*?)\)\)\) ⊕ (\w+\([^()]*\))"
    def replace_special_xor(match):
        return ("(((" + match.group(1) + ")) & -" + match.group(2) + ") | (-(" + match.group(1) + ")) & " + match.group(2) + ")")
    logic_formula = re.sub(special_xor_pattern, replace_special_xor, logic_formula)
    return logic_formula


# import re

import re
from collections import defaultdict

def sanitize_fol(s: str) -> str:
    """
    1) Convert Unicode symbols and numerals to ASCII/NLTK form
    2) Normalize reverse arrows `<-` to `(...) -> (...)`
    3) Protect decimal dots in tokens like '42.3billion' -> '42Dot3billion'
    """
    def _unquote(match):
        inner = match.group(1)
        # remove non‐alphanumeric, then capitalize words and join
        parts = re.findall(r"[A-Za-z0-9]+", inner)
        return "".join(w.capitalize() for w in parts)
    
    s = re.sub(r'"([^"]+)"', _unquote, s)

    # 1) Unicode / simple replacements
    translation_map = {
        "∀": "all",
        "All": "all",
        "∃": "exists",
        "Exists": "exists",
        "→": "->",
        "¬": "-",
        "~": "-"
        
    #     "∧": "&",
    #     "∨": "|",
    #     "⟷": "<->",
    #     "↔": "<->",
    #     # Note: we’ll handle “.” below for numerics, so remove from here
    #     "Ś": "S",
    #     "ą": "a",
    #     "’": "",
    #     "! ": "",
    #     "!": "",
    }
    
    for u, ascii_ in translation_map.items():
        s = s.replace(u, ascii_)
   
   

    def protect_decimal_tokens(formula: str) -> str:
        return re.sub(
            r"(\d+)\.(\d+)([A-Za-z]\w*)",
            lambda m: f"{m.group(1)}Dot{m.group(2)}{m.group(3)}",
            formula
        )

    def normalize_comparisons(formula: str) -> str:
        # Handle <= and >= first
        formula = re.sub(
            r"(\b[A-Za-z][A-Za-z0-9_]*\b)\s*<=\s*([A-Za-z0-9_'.]+)",
            r"LesserThanOrEqual(\1,\2)", formula
        )
        formula = re.sub(
            r"(\b[A-Za-z][A-Za-z0-9_]*\b)\s*>=\s*([A-Za-z0-9_'.]+)",
            r"GreaterThanOrEqual(\1,\2)", formula
        )
        # Now standard <, >, =
        formula = re.sub(
            r"(\b[A-Za-z][A-Za-z0-9_]*\b)\s*<\s*([A-Za-z0-9_'.]+)",
            r"LesserThan(\1,\2)", formula
        )
        formula = re.sub(
            r"(\b[A-Za-z][A-Za-z0-9_]*\b)\s*>\s*([A-Za-z0-9_'.]+)",
            r"GreaterThan(\1,\2)", formula
        )
        formula = re.sub(
            r"(\b[A-Za-z][A-Za-z0-9_]*\b)\s*=\s*([A-Za-z0-9_'.]+)",
            r"Equal(\1,\2)", formula
        )
        return formula

    def normalize_biconditional(formula: str) -> str:
        """
        Convert all occurrences of (A <-> B) into ((A -> B) & (B -> A)).
        """
        # non-greedy match inside parentheses
        pattern = r"\(([^()]*?)\s*<->\s*([^()]*?)\)"
        while re.search(pattern, formula):
            formula = re.sub(pattern, r"((\1 -> \2) & (\2 -> \1))", formula)
        return formula
    
    def fix_misparsed_negations(formula: str) -> str:
        """
        Convert '(> ANY)' back to '-(ANY)', supporting compound expressions inside.
        """
        # match '(> <content>)' and replace with '-(content)'
        return re.sub(r"\(>\s*([^)]+)\)", r"-(\1)", formula)
   
   
    s = re.sub(
        r'(\d+)\.(\d+)([A-Za-z]\w*)',
        lambda m: f"{m.group(1)}Dot{m.group(2)}{m.group(3)}",
        s
    )
    
    # 5) Remove spurious dots after variables: “x.y” → “xy”
    s = re.sub(r"([a-z])\.(?=[a-z])", lambda m: m.group(1), s)

    # 6) Normalize reverse-implication arrows
    s = re.sub(
        r'(.+?)\s*<-\s*(.+)',
        lambda m: f"({m.group(2).strip()} -> {m.group(1).strip()})",
        s
    )
    
   
    s = re.sub(r'"([^"]+)"', _unquote, s)
    s =  normalize_biconditional(s)
    s = fix_misparsed_negations(s)
    s = normalize_comparisons(s)
    s = protect_decimal_tokens(s)
    
    return s

def evaluate_math():
    pass

def evaluate(premises, conclusion, save_dir=False):
    # all_forms = premises + [conclusion]
    # all_forms = disambiguate_arities(all_forms)

    # # split them back out
    # premises, conclusion = all_forms[:-1], all_forms[-1]
    
    premises = [sanitize_fol(p) for p in premises]
    conclusion = sanitize_fol(conclusion)
    # os.path.join(save_dir)
    # Create error directory if it doesn't exist
    # error_dir = "prover_errors"
    # os.makedirs(error_dir, exist_ok=True)
    
    try:
        c = read_expr(conclusion)
        p_list = [read_expr(p) for p in premises]
        
        # Attempt proof
        truth_value = prover.prove(c, p_list)
        
        if truth_value:
            return "True"
        else:
            # Attempt negation proof
            neg_c = read_expr("-(" + conclusion + ")")
            negation_true = prover.prove(neg_c, p_list)
            
            if negation_true:
                return "False"
            else:
                return "Uncertain"
                
    except Exception as e:
        # Save error details
        error_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "premises": premises,
            "conclusion": conclusion,
            "error_type": str(type(e)),
            "error_message": str(e),
            "stack_trace": traceback.format_exc()
        }
        if save_dir:
            exp_dir = os.path.join("results", save_dir)
            # Append to error log file
            error_file = os.path.join(exp_dir, "prover_errors.jsonl")
            with open(error_file, "a") as f:
                f.write(json.dumps(error_data) + "\n")
        # print(error_data)
        return error_data
        
