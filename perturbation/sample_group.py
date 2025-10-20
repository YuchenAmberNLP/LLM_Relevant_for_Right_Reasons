import json
from pathlib import Path
import argparse

def ngram_overlap(query, passage, n=1):
    query_tokens = query.lower().split()
    passage_tokens = passage.lower().split()
    query_ngrams = [' '.join(query_tokens[i:i+n]) for i in range(len(query_tokens)-n+1)]
    passage_ngrams = set([' '.join(passage_tokens[i:i+n]) for i in range(len(passage_tokens)-n+1)])
    return sum(1 for ng in query_ngrams if ng in passage_ngrams)

def extract_conflict_samples(input_path, lower=0, upper=None, output_dir="data/classification_range"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_stem = input_path.stem
    lang_pair = file_stem.split('_')[0]

    if upper is None:
        range_str = f"ge{lower}"
    else:
        range_str = f"ge{lower}_le{upper}"

    output_conflict = output_dir / f"{lang_pair}_conflict_irrel_{range_str}.jsonl"

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_conflict, 'w', encoding='utf-8') as conflict_out:

        for line in infile:
            ex = json.loads(line)
            query = ex['query']
            passage_A = ex['passage_A']['passage']
            passage_B = ex['passage_B']['passage']
            label = ex['output'].strip()

            rel_doc = passage_A if label == "Passage A" else passage_B
            irrel_doc = passage_B if label == "Passage A" else passage_A

            rel_overlap = ngram_overlap(query, rel_doc)
            irrel_overlap = ngram_overlap(query, irrel_doc)

            if irrel_overlap >= lower and (upper is None or irrel_overlap <= upper):
                if rel_overlap == 0:
                    json.dump(ex, conflict_out, ensure_ascii=False)
                    conflict_out.write("\n")

    return str(output_conflict)

def main():
    parser = argparse.ArgumentParser(description="Extract conflict samples based on irrel overlap range.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--lower", type=int, default=0, help="Lower bound of irrel overlap")
    parser.add_argument("--upper", type=str, default=None, help="Upper bound of irrel overlap (or 'None')")
    parser.add_argument("--output_dir", type=str, default="data/classification_range", help="Directory to save output")

    args = parser.parse_args()

    # Convert upper from string to int or None
    upper = None if args.upper in [None, "None", ""] else int(args.upper)

    conflict_file = extract_conflict_samples(
        args.input_file, lower=args.lower, upper=upper, output_dir=args.output_dir
    )
    print(f"Saved conflict samples to: {conflict_file}")

if __name__ == "__main__":
    main()
