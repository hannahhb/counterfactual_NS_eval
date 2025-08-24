import re
from nltk.sem import Expression
from nltk.inference import Prover9

# 1) Paste in your full <EVALUATE> section here as a raw string
file_path = '/data/projects/punim0478/bansaab/linc2/data/train.txt'

with open(file_path, 'r') as file:
    fols = file.read()

evaluate_blocks = re.findall(
    r'<EVALUATE>\s*(.*?)\s*</EVALUATE>',
    fols,
    flags=re.DOTALL
)

# print(evaluate_blocks)
read_expr = Expression.fromstring

for i, block in enumerate(evaluate_blocks, start=1):
    # 1) extract every “FOL: …” line
    fol_lines = re.findall(
        r'FOL:\s*(.+?)\r?\\n', 
        block
    )

    if not fol_lines:
        print(f"[Block {i}] no FOL: lines found, skipping")
        continue

    # 2) last line is the conclusion, the rest are premises
    premises_strs = fol_lines[:-1]
    conclusion_str = fol_lines[-1]

    # 3) parse
    try:
        premises = [read_expr(s) for s in premises_strs]
        goal     = read_expr(conclusion_str)
    except Exception as e:
        print(f"[Block {i}] parse error: {e!r}")
        continue

    # 4) run Prover9
    prover = Prover9(timeout=10)
    print(f"\n=== Block {i} ===")
    print(len(fol_lines))

    for p in premises:
        print("  premise:", p)
    print("  goal:   ", goal)

    try:
        if prover.prove(goal, premises):
            print("✅ conclusion follows (True)")
        else:
            # try the negation
            neg = read_expr(f"-({conclusion_str})")
            if prover.prove(neg, premises):
                print("❌ negation follows (so conclusion is False)")
            else:
                print("❓ neither provable (Uncertain)")
    except Exception as e:
        print(f"⚠️ Prover9 error: {e!r}")