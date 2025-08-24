import re, json, sys
from pathlib import Path

EXAMPLE_RE = re.compile(
    r"=== Example (?P<idx>\d+) ===\n"
    r"Prompt:\n(?P<prompt>.+?)\n\n"
    r"(?P<body>.+?)(?=(?:=== Example \d+ ===|$))",
    re.S
)
# capture each Generation block including all lines up to the next Generation or end
GEN_RE = re.compile(
    r"(Generation\s+\d+:\n"          # the Generation header
    r"(?:^(?!Generation\s+\d+:).*\n?)+)",  # all following lines not starting with "Generation N:"
    re.M
    
)
ANS_RE = re.compile(r"Processed:\s*(?P<ans>\w+)")
REF_RE = re.compile(r"Reference:\s*(?P<ref>\w+)")
PROMPT_END_MARK = re.compile(r"</EVALUATE>")

def parse_examples(text):
    logs = []
    progress = {}
    for m in EXAMPLE_RE.finditer(text):
        idx = int(m.group("idx"))
        block = m.group("body")
        # extract generations
        gens = GEN_RE.findall(block)
        gens = [g.strip() for g in GEN_RE.findall(block)]
        # extract processed answers
        answers = ANS_RE.findall(block)
        # extract reference
        ref_m = REF_RE.search(block)
        reference = ref_m.group("ref") if ref_m else None
        # reconstruct prompt (everything from prompt up through the closing </EVALUATE>)
        prompt = m.group("prompt")
        # body up to first Generation
        body_until_eval = block.split("Generation",1)[0]
        # ensure we include the <EVALUATE> section
        if not PROMPT_END_MARK.search(body_until_eval):
            # fallback: include whole body until first "Answer:"
            body_until_eval = block.split("Answer:",1)[0] + "Answer:"
        full_prompt = prompt + "\n\n" + body_until_eval

        logs.append({
            "idx": idx,
            "prompt": full_prompt.strip(),
            "raw_generations": gens,
            "answers": answers
        })

        progress[str(idx)] = {
            "generations": answers,
            "reference": reference
        }

    return logs, progress

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_logs.py chat_dump.txt")
        sys.exit(1)
    dump_path = sys.argv[1]
    log_path = dump_path.replace("log.txt", "log.jsonl")
    prog_path = dump_path.replace("log.txt", "progress.json")
    log_path, prog_path = Path(log_path),  Path(prog_path)
    dump_path = Path(dump_path).read_text(encoding="utf-8")
    

    logs, progress = parse_examples(dump_path)

    # write log.jsonl
    with log_path.open("w", encoding="utf-8") as f:
        for rec in sorted(logs, key=lambda x: x["idx"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # write progress.json
    with prog_path.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(logs)} examples → {log_path} and {prog_path}")
