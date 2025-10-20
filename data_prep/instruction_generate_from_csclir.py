# https://github.com/spacemanidol/MSMARCO/blob/master/Ranking/README.md
import sys
import json
import random
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import os
from ftfy import fix_encoding
import random
import argparse

def generate_jsonl_pairs(input_file):
    qid_to_samples = {}

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            data = json.loads(line.strip())
            qid = data["qid"]
            query = fix_encoding(data["query"])
            pid = data["pid"]
            passage = fix_encoding(data["passage"])
            label = data["label"]
            # print(passage)

            if qid not in qid_to_samples:
                qid_to_samples[qid] = {
                    "query": query,
                    "positive": [],
                    "negative": [],
                    "seen_positives": set(),
                    "seen_negatives": set()
                }

            if label == 1 and pid not in qid_to_samples[qid]["seen_positives"]:
                qid_to_samples[qid]["positive"].append({"pid": pid, "passage": passage, "label": 1})
                qid_to_samples[qid]["seen_positives"].add(pid)

            elif label == 0 and pid not in qid_to_samples[qid]["seen_negatives"]:
                qid_to_samples[qid]["negative"].append({"pid": pid, "passage": passage, "label": 0})
                qid_to_samples[qid]["seen_negatives"].add(pid)

    # print(qid_to_samples)
    return qid_to_samples

def generate_instructions(input_file, output_dir, max_len=None, neg_samples_ratio=None):
    qid_to_samples = generate_jsonl_pairs(input_file)
    count = 0
    examples = []
    output_dir = Path(output_dir)
    input_path = Path(input_file).resolve()
    
    # Create output directory structure based on input file path
    try:
        # Find the index of 'src/data' in the path
        path_parts = input_path.parts
        src_data_idx = path_parts.index('src') + 2  # +2 to get past 'src/data'
        # Get the subdirectory structure after 'src/data'
        subdir_structure = Path(*path_parts[src_data_idx:-1])  # Exclude the filename
        # Create the full output path
        output_path = output_dir / subdir_structure
        output_path.mkdir(parents=True, exist_ok=True)
        
        # # Original instruction file name generation
        # stem = input_path.stem
        # instruction_file = output_path / f"instruction_mix_{stem}.json"
        
        # New instruction file name generation
        # Extract language code from path (e.g., 'xxxx' from the path)
        lang_code = path_parts[-2]  # Get the language code from the directory name
        instruction_file = output_path / f"instruction_mix_train_{lang_code}squad.json"
    except ValueError:
        # If 'src/data' is not in the path, use the original logic
        stem = input_path.stem
        instruction_file = output_dir / f"instruction_mix_{stem}.json"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)

    for qid, samples in qid_to_samples.items():
        query = samples["query"]
        positives = samples["positive"]
        negatives = samples["negative"]

        if not positives or not negatives:
            continue

        for pos_sample in positives:
            if neg_samples_ratio is None:
                selected_negatives = negatives
            else:
                # limit negtive sample ratio
                num_samples = min(neg_samples_ratio, len(negatives))
                selected_negatives = random.sample(negatives, num_samples)

            for neg_sample in selected_negatives:
                if random.random() < 0.5:
                    doc1, doc2 = pos_sample, neg_sample
                    answer = "Passage A"
                else:
                    doc1, doc2 = neg_sample, pos_sample
                    answer = "Passage B"
                alpaca_example = {
                    "instruction": """You are an expert in multilingual information retrieval. Your task is to determine which of the two passages is more relevant to the given query.\n\nStrict instructions:\n- Do NOT provide any explanation.\n- Do NOT include any additional words, punctuation, or formatting.\n- Answer with only \"Passage A\" or \"Passage B\" (without quotes).""",
                    "input": f"**Query**: {query}\n\n**Passage A**: {doc1['passage']}\n\n**Passage B**: {doc2['passage']}\n\nWhich passage is more relevant to the query? Respond with exactly one of the following options:\nPassage A\nPassage B\n\nYour answer:",
                    "output": answer
                }

                examples.append(alpaca_example)
                count += 1
                if max_len is not None and count >= max_len:
                    print(f"Reached max_len limit: {max_len} pairs generated.")
                    break

            if max_len is not None and count >= max_len:
                break

        if max_len is not None and count >= max_len:
            break

    with open(instruction_file, "w", encoding="utf-8") as fout:
        json.dump(examples, fout, ensure_ascii=False, indent=2)

    print(f"Successfully generated {count} pairs in instruction file: {instruction_file}")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate Alpaca-style instruction data from passage pairs.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input .jsonl file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the instruction JSON file.")
    parser.add_argument("--max_len", type=int, default=None, help="Maximum number of examples to generate.")
    parser.add_argument("--neg_samples_ratio", type=int, default=None, help="Number of negative samples per positive passage.")
    
    args = parser.parse_args()

    generate_instructions(
        input_file=args.input_path,
        output_dir=args.output_dir,
        max_len=args.max_len,
        neg_samples_ratio=args.neg_samples_ratio
    )



