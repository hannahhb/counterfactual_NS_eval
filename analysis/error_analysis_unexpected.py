#!/usr/bin/env python3
import json
import re
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    print("Usage: python analyze_unexpected_token_errors.py path/to/log.jsonl")
    sys.exit(1)

LOG_PATH = sys.argv[1]

# We'll group by the exact “Unexpected token: 'XYZ'” string
error_groups = defaultdict(list)

pattern = re.compile(r"Unexpected token: '([^']+)'")

with open(LOG_PATH) as f:
    for lineno, line in enumerate(f, 1):
        entry = json.loads(line)
        err = entry.get("error_message","")
        m = pattern.search(err)
        if m:
            token = m.group(1)
            # capture a little context: premises, conclusion and the message
            context = {
                "idx": entry.get("idx", None),
                "error_message": err,
                "premises": entry.get("premises", []),
                "conclusion": entry.get("conclusion", "")
            }
            error_groups[token].append(context)

# Print summary
print(f"Found {sum(len(v) for v in error_groups.values())} 'Unexpected token' errors\n")
for token, contexts in error_groups.items():
    print(f"Token `{token}`   — {len(contexts)} occurrences")
    # show up to 3 examples
    for ex in contexts[:3]:
        print(f"  • idx={ex['idx']}, conclusion=`{ex['conclusion'][:60]}…`")
    print()

# Optionally write out a JSON summary
SUMMARY_PATH = LOG_PATH.replace(".jsonl", "_unexpected_summary.json")
with open(SUMMARY_PATH, "w") as out:
    json.dump({tok: len(ctxs) for tok,ctxs in error_groups.items()}, out, indent=2)

print(f"Summary counts by token written to {SUMMARY_PATH}")
