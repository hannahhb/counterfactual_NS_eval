import csv

correct = 0
total = 0

with open('results.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        predicted_tag = row[-2]  # second-last column
        actual_tag = row[-1]     # last column
        if predicted_tag == actual_tag:
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.2%}")
