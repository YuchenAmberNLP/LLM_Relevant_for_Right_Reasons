

from pairwise_reranker_Llama3 import PairwiseLlmRanker_HF, get_overlap_group
from pairwise_reranker_vicuna import PairwiseLlmRanker_vicuna
from typing import List, Dict
from llmrankers.rankers import LlmRanker, SearchResult
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
import copy
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import re
import json
import os
import logging
import random
import argparse
from pathlib import Path
import ftfy


def setup_logging(model_name, input_path_short):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


    if not logger.handlers:
        file_handler = logging.FileHandler(f"logs/classification_{model_name}_{input_path_short}.log", mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)


def clean_text(text: str) -> str:
    """Clean text using ftfy to fix encoding issues."""
    return ftfy.fix_text(text)



def classify_samples(reranker, jsonl_path, model_name, max_eval=None, output_root="causal_results"):
    """New function for simple classification and error collection"""
    json_path = Path(jsonl_path)

    model_path = Path(model_name)
    if model_path.exists():
       model_name_short=model_path.parent.name
    else:
        model_name_short=model_name.split("/")[-1]

    parts = [p.lower() for p in json_path.parts]
    split_name = "original" if "original" in parts else "perturbed" if "perturbed" in parts else "unspecified"

    output_dir = Path(output_root) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)


    results_path  = output_dir / f"{json_path.stem}_{model_name_short}_results.jsonl"
    errors_path   = output_dir / f"{json_path.stem}_{model_name_short}_errors.jsonl"
    corrects_path = output_dir / f"{json_path.stem}_{model_name_short}_corrects.jsonl"

    
    total = 0
    correct = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f, \
         open(results_path, 'w', encoding='utf-8') as rf, \
         open(errors_path, 'w', encoding='utf-8') as ef, \
         open(corrects_path, 'w', encoding='utf-8') as cf:
        
        for line in f:
            if max_eval and total >= max_eval:
                break

            item = json.loads(line)
            # Clean text using ftfy
            query = clean_text(item["query"])
            passage_A = clean_text(item["passage_A"]["passage"])
            passage_B = clean_text(item["passage_B"]["passage"])
            label = clean_text(item.get("gold_output", item.get("output", "")))
            qid = item["qid"]

            prediction = reranker.compare(query=query, docs=[passage_A, passage_B])
            
            # Save all results with cleaned text
            result = {
                "qid": qid,
                "query": query,  # cleaned query
                "passage_A": {
                    "pid": item["passage_A"]["pid"],
                    "passage": passage_A  # cleaned passage
                },
                "passage_B": {
                    "pid": item["passage_B"]["pid"],
                    "passage": passage_B  # cleaned passage
                },
                "gold_output": label,  # cleaned label
                "model_output": prediction
            }
            json.dump(result, rf, ensure_ascii=False); rf.write("\n")

            if prediction.strip() == label.strip():
                json.dump(result, cf, ensure_ascii=False); cf.write("\n")
                correct += 1
            else:
                json.dump(result, ef, ensure_ascii=False); ef.write("\n")
            total += 1

            if total % 100 == 0:
                reranker.logger.info(f"Processed {total}, accuracy: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    reranker.logger.info(f"Final accuracy: {accuracy} ({correct}/{total})")
    reranker.logger.info(f"Results saved to: {results_path}")
    reranker.logger.info(f"Error samples saved to: {errors_path}")
    reranker.logger.info(f"Correct samples saved to: {corrects_path}")
    summary_path = output_dir / "summary_log.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"[{model_name_short}] on {json_path.name} → Accuracy: {accuracy} ({correct}/{total})\n")
    
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="LLM model name")
    parser.add_argument("--max_eval", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_root", type=str, default="causal_results",
                        help="Root folder to save outputs (default: causal_results). "
                             "Outputs will be placed under <output_root>/<original|perturbed>/")

    args = parser.parse_args()

    model_name_short = args.model_name.split("/")[-1]
    input_path_short = Path(args.input_path).stem
    setup_logging(model_name_short, input_path_short)
    logger = logging.getLogger()

    generate_params = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.85
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reranker = PairwiseLlmRanker_HF(
        logger=logger,
        generate_params=generate_params,
        model_name_or_path=args.model_name,
        device=device
    )
    
    result = classify_samples(reranker, args.input_path, model_name=args.model_name, max_eval=args.max_eval, output_root=args.output_root)
    logger.info(f"Evaluation on {args.input_path} with model {args.model_name}")
    logger.info(f"Classification accuracy: {result}")


