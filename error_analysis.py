import argparse
from collections import Counter
import re
import json

def classify_error(message):
    if 'multiple arities' in message:
        return 'multiple Arity Error'
    elif 'Unexpected token' in message:
        # print(message)
        return 'Unexpected Token'
    # elif 'unexpected end of input' in message or 'EOF' in message:
    #     print(message)
    #     return 'End of Input (Incomplete)'
    elif 'cannot be constructed from the marked string' in message:
        return 'Unconstructable Term'
    elif "End of input found" in message:
        # print(message)
        return "End of input found Error"
    elif "symbols/arities are used as both relation and function" in message:
        return "used as both relation and function"
    elif "not a Lambda Expression" in message:
        return "not a Lambda Expression"
    # elif 'Prover9FatalException' in message:
    #     return 'Prover9 Fatal Error'
    # elif 'LogicalExpressionException' in message:
    #     return 'Parsing LogicException'
    elif """'utf-8' codec can't decode byte""" in message:
        return "utf-8 codec error"
    elif "The following symbols cannot be used as atomic formulas, because they are variables": 
        return "variable cannot be atomic formula"
    elif "MAX_SECONDS" in message:
        return "MAX_SECONDS error"
    else:
        print(message)
        return 'Other Error'

def analyze_errors(error_logs):
    error_categories = []
    for log in error_logs:
        msg = log.get('error_message', '') 
        error_categories.append(classify_error(msg))
    return Counter(error_categories)

def main():
    parser = argparse.ArgumentParser(description='Analyze logical inference errors.')
    parser.add_argument('input_file', help='Path to the JSONL file with error logs')
    args = parser.parse_args()

    with open(args.input_file) as f:
        logs = [json.loads(line) for line in f]

    error_counts = analyze_errors(logs)
    for error, count in error_counts.items():
        print(f"{error}: {count}")

if __name__ == "__main__":
    main()
