#!/usr/bin/env python3
import sys, json, io, os

def dedup_by_idx(in_path: str, out_path: str):
    seen = set()
    kept, skipped, bad = 0, 0, 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue

            if "idx" not in obj:
                # if you want to keep no-idx lines, write them:
                # fout.write(line + "\n"); kept += 1
                bad += 1
                continue

            idx = obj["idx"]
            if idx in seen:
                # duplicate -> drop (keep the first instance already written)
                skipped += 1
                continue

            seen.add(idx)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Input:   {in_path}")
    print(f"Output:  {out_path}")
    print(f"Kept:    {kept}")
    print(f"Skipped (dupes): {skipped}")
    print(f"Bad/invalid lines: {bad}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dedup_log.py <log.jsonl> [out.jsonl]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(in_path)[0] + "_dedup.jsonl"
    dedup_by_idx(in_path, out_path)
