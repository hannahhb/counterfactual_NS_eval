import re
from collections import defaultdict

import sys, os

# 1) Compute the absolute path to the folder above `linc2/`
sys.path.insert(0, "/data/projects/punim0478/bansaab/linc2/")

from eval.tasks.utils import evaluate

def find_arities(formula: str) -> dict[str, set[int]]:
    # existing code
    arities = defaultdict(set)
    if not isinstance(formula, str):
        return arities
    for name, args in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*\(([^)]*)\)", formula):
        args = args.strip()
        if args == '': arities[name].add(0)
        else: arities[name].add(len([a for a in re.split(r'\s*,\s*', args) if a]))
    for name in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b(?!\s*\()", formula):
        arities[name].add(0)
    return arities


def resolve_multiple_arities(formulas: list[str], var: str = 'x') -> list[str]:
    # existing arity resolution
    global_ar = defaultdict(set)
    for f in formulas:
        for name, ar in find_arities(f).items(): global_ar[name].update(ar)
    to_fix = {n for n, ar in global_ar.items() if 0 in ar and len(ar) > 1}
    new = []
    for f in formulas:
        if not isinstance(f, str): new.append(f); continue
        m = f
        if any(re.search(rf"\b{name}(?:\(\)|\b(?!\s*\())", m) for name in to_fix):
            if not m.strip().startswith(f"all {var}."): m = f"all {var}. {m}"
            for name in to_fix:
                m = re.sub(rf"\b{name}\(\)", f"{name}({var})", m)
                m = re.sub(rf"\b{name}\b(?!\s*\()", f"{name}({var})", m)
        new.append(m)
    return new


def find_symbol_roles(formulas: list[str]) -> dict[str, dict[str, bool]]:
    """
    Determine for each symbol whether it's used as a predicate or as a function.
    Returns dict: symbol -> {'predicate': bool, 'function': bool}.
    """
    roles = defaultdict(lambda: {'predicate': False, 'function': False})
    for f in formulas:
        if not isinstance(f, str): continue
        # predicate usage: symbol appears in head or as boolean test
        for name, _ in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*\(", f):
            roles[name]['predicate'] = True
        # function usage: symbol appears in argument position inside another predicate
        # approximate by searching nested parentheses
        for func_call in re.findall(r"\w+\([^()]*\)", f):
            inner = re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*\(", func_call)
            for name in inner:
                # if entire formula is exactly name(...) skip
                if not re.fullmatch(rf"{name}\([^()]*\)", f.strip()):
                    roles[name]['function'] = True
    return roles


def resolve_predicate_function_conflicts(formulas: list[str]) -> list[str]:
    """
    For any symbol used both as predicate and function, rename predicate occurrences by prefixing 'Is'.
    """
    roles = find_symbol_roles(formulas)
    conflicts = [n for n, r in roles.items() if r['predicate'] and r['function']]
    new = []
    for f in formulas:
        if not isinstance(f, str): new.append(f); continue
        m = f
        for name in conflicts:
            # rename predicate calls 'Name(' to 'IsName('
            m = re.sub(rf"\b{name}\(", f"Is{name}(", m)
            # also rename bare predicate in head positions
            m = re.sub(rf"\b{name}\b", f"Is{name}", m)
        new.append(m)
    return new


# Example usage:
premises = ["InfectiousDisease(d) & CausedBy(Virus(d), MonkeypoxVirus)", 
            "OccursIn(Virus(Monkeypox), Animals) & In(Animals, Humans)", 
            "all x. (Humans(x) -> Mammal(x))", 
            "all x. (Mammal(x) -> Animals)", 
            "-InfectiousDisease(d) -> FeelTired(Humans)", 
            "In(Humans, glu) -> FeelTired(Humans)"]

conclusion = "exists x. Animals(x)"

# rename Animals/1 predicate to IsAnimal/1
premises = ["InfectiousDisease(d) & CausedBy(Virus(d), MonkeypoxVirus)", 
            "OccursIn(Virus(Monkeypox), Animals) & In(Animals, Humans)", 
            "all x. (Humans(x) -> Mammal(x))", 
            "all x. (Mammal(x) -> Animals(x))", 
            "-InfectiousDisease(d) -> FeelTired(Humans)", 
            "In(Humans, glu) -> FeelTired(Humans)"]

conclusion = "exists x. Animals(x)"
print(find_arities("InfectiousDisease(d) & CausedBy(Virus(d), MonkeypoxVirus)"))
# cleaned = resolve_multiple_arities(premises)
# print(evaluate(premises, conclusion))  # ['all x. Human(x)', 'all x. (Human(x) -> Mammal(x))']


""""
TEXT: Monkeypox is an infectious disease caused by the monkeypox virus.\nREASONING: This establishes the disease and its causative agent.\nFOL: InfectiousDisease(Monkeypox) & CausedBy(Monkeypox, MonkeypoxVirus)
\nTEXT: Monkeypox virus can occur in certain animals, including humans.\nREASONING: The virus can infect animals, including humans.\nFOL: CanOccurIn(MonkeypoxVirus, Animals) & CanOccurIn(MonkeypoxVirus, Humans)
\nTEXT: Humans are mammals.\nREASONING: A taxonomic relationship is stated.\nFOL: Mammal(Humans)
\nTEXT: Mammals are animals.\nREASONING: A broader taxonomic relationship.\nFOL: Animal(Mammals)
\nTEXT: Symptons of Monkeypox include fever, headache, muscle pains, feeling tired, and so on.\nREASONING: Symptoms are listed, although not formalized in FOL.
\nTEXT: People feel tired when they get a glu.\nREASONING: A conditional statement about people and fatigue.\nFOL: all x. (Person(x) & GetGlu(x) -> FeelTired(x))
\nTEXT: There is an animal.\nREASONING: We verify that the premises imply the existence of an animal.\nFOL: exists x. Animal(x)" this example was right what can we learn from this

"""