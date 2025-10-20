import json
import random
import argparse

def _overlap_by_space(a, b):

    ta = set(a.lower().split())
    tb = set(b.lower().split())
    return len(ta & tb) > 0

def _check_condition(sample, condition):
    q = sample.get("query", "")
    a_pass = sample.get("passage_A", {}).get("passage", "")
    b_pass = sample.get("passage_B", {}).get("passage", "")
    a_overlap = _overlap_by_space(q, a_pass)
    b_overlap = _overlap_by_space(q, b_pass)

    if condition == "conflict":
        cond_ok = (b_overlap and not a_overlap)
    elif condition == "non_conflict":
        cond_ok = (a_overlap and not b_overlap)
    else:
        raise ValueError("condition must be 'conflict' or 'non_conflict'")

    return cond_ok, a_overlap, b_overlap

def process_jsonl(input_path, output_path, seed):
    if seed is not None:
        random.seed(seed)

    saved = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            cond, a_overlap, b_overlap = _check_condition(sample)
            if not cond:
                print(f"[Line {line_no}] wrong overlapping tokens: A_overlap={a_overlap}, B_overlap={b_overlap}")
                print(sample)
                continue

            swap = random.random() < 0.5
            if swap:
                sample["passage_A"], sample["passage_B"] = sample["passage_B"], sample["passage_A"]
                sample["output"] = "Passage B" 
            else:
                sample["output"] = "Passage A"

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            saved += 1

    print(f"saved samples: {saved}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and label synthetic causal data by overlap condition.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument("--condition", required=True, choices=["conflict", "non_conflict"],
                        help="Overlap condition to enforce: 'conflict' or 'non_conflict'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for swapping A/B.")
    args = parser.parse_args()

    process_jsonl(args.input, args.output, condition=args.condition, seed=args.seed)