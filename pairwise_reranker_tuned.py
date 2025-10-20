# -*- coding: utf-8 -*-
"""
Modified from: https://github.com/ielab/llm-rankers/blob/main/llmrankers/pairwise.py

This script is adapted and extended by Yuchen Mao as part of the paper "Relevant for the Right Reasons? Investigating Lexical Biases in Re-ranking with Large Language Models".

Original authors: University of Queensland IELab
For academic and non-commercial use only.
"""
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
import transformers
# from vllm import LLM, SamplingParams
from multiprocessing import Pool
import math
from functools import partial

import random
import numpy as np
import torch
import multiprocessing as mp
import signal
import sys
import traceback
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_overlap_group(query, relevant_passage):
    query_tokens = set(query.lower().split())
    passage_tokens = set(relevant_passage.lower().split())
    overlap = len(query_tokens & passage_tokens)
    if overlap == 0:
        return "no_overlap"
    elif 1 <= overlap <= 3:
        return "some_overlap"
    else:
        return "significant_overlap"

def setup_logging(log_path):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger = logging.getLogger()

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)



class PairwiseLlmRanker_HF(LlmRanker):
    def __init__(self,
                 logger,
                 generate_params,
                 model_name_or_path="meta-llama/Llama-3.2-1B",
                 method="sliding",
                 batch_size=2,
                 k=10,
                 device=None):
        self.model_name_or_path = model_name_or_path
        self.logger = logger
        self.k = k
        self.method = method
        self.batch_size = batch_size
        self.total_compare = 0
        self.total_response_count = 0
        self.strict_match_count = 0
        self.approx_match_count = 0
        self.generate_params = generate_params

        self.CHARACTERS = ["A", "B"]
        self.system_prompt = """
            You are an expert in multilingual information retrieval. Your task is to determine which of the two passages is more relevant to the given query.

            Strict instructions:
            - Do NOT provide any explanation.
            - Do NOT include any additional words, punctuation, or formatting.
            - Answer with only Passage A or Passage B (without quotes).  
            """
        self.user_prompt = """
            **Query**: {query}
            
            **Passage A**: {doc1}
            
            **Passage B**: {doc2}
            
            Which passage is more relevant to the query? Respond with exactly one of the following options: 
            Passage A 
            Passage B
            
            Your answer:
            """ #important


        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        # use multi gpus
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True).to(self.device)
        special_tokens_dict = {}
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=self.tokenizer,
            model=self.model,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.generate_params["eos_token_id"] = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id


    def extract_passage_choice(self, decoded_response):
        first_line = decoded_response.strip().split("\n")[0]
        matches = re.findall(r"\bPassage [AB]\b", first_line)
        if len(matches) == 1:
            return matches[0]
        else:
            return "Fail"

    def _get_response(self, input_text):
        message = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        formatted_input = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_input, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            # print("Response generating...")
            output_tokens = self.model.generate(**inputs, pad_token_id=self.tokenizer.pad_token_id, **self.generate_params)

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output_tokens[0][input_length:]
        decoded_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


        format_answer = self.extract_passage_choice(decoded_response)


        answer_lower = decoded_response.lower()
        format_answer_lower = format_answer.lower()

        if answer_lower in ["passage a", "passage b"]:
            strict_match = True
            approx_match = True
            output = decoded_response[-1].upper()
        elif format_answer_lower in ["passage a", "passage b"]:
            strict_match = False
            approx_match = True
            self.logger.info(f"Unexpected output: {decoded_response}")
            output = format_answer[-1].upper()
        else:
            strict_match = False
            approx_match = False
            self.logger.info(f"Unexpected formatted output: {decoded_response}")
            output = "A" #default

        self.total_response_count += 1
        if strict_match:
            self.strict_match_count += 1
        if approx_match:
            self.approx_match_count += 1
        if self.total_response_count % 1000 == 0:
            self.logger.info(f"Strict match rate: {self.strict_match_count}/{self.total_response_count}")
            self.logger.info(f"Approximate match rate: {self.approx_match_count}/{self.total_response_count}")

        return output

    def compare(self, query: str, docs: List, bidirection=False):
        self.total_compare += 1
        doc1, doc2 = docs[0], docs[1]
        input_texts = [self.user_prompt.format(query=query, doc1=doc1, doc2=doc2),
                       self.user_prompt.format(query=query, doc1=doc2, doc2=doc1)]
        # bidirection comparison
        if bidirection:
            return [f'Passage {self._get_response(input_texts[0])}', f'Passage {self._get_response(input_texts[1])}']
        else:
            return f'Passage {self._get_response(input_texts[0])}'



    def rerank(self, qid, query: str, ranking: List, bidirection=False):
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        if self.method == "sliding":
            # k = 10
            k = min(self.k, len(ranking))

            last_end = len(ranking) - 1
            for i in range(k):
                current_ind = last_end
                is_change = False
                while True:
                    if current_ind <= i:
                        break
                    doc1 = ranking[current_ind]
                    doc2 = ranking[current_ind - 1]
                    # use bidirection:
                    output = self.compare(query, [doc1["text"], doc2["text"]], bidirection)
                    if (bidirection and (output[0] == "Passage A" and output[1] == "Passage B")) or (not bidirection and output == "Passage A"):
                        ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]

                        if not is_change:
                            is_change = True
                            if last_end != len(ranking) - 1:  # skip unchanged pairs at the bottom
                                last_end += 1
                    if not is_change:
                        last_end -= 1
                    current_ind -= 1

        results = []
        top_doc_ids = set()
        rank = 1
        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc["doc_id"])
            results.append({
                "query_id": qid,
                "doc_id": doc["doc_id"],
                "score": float(self.k - rank + 1)
            })
            rank += 1
        for doc in original_ranking:
            if doc["doc_id"] not in top_doc_ids:
                results.append({
                    "query_id": qid,
                    "doc_id": doc["doc_id"],
                    "score": 0.0
                })
                rank += 1
        return results



def load_msmarco_topn(input_path):
    qid2entry = {}
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                try:
                    qid, pid, query, passage = line.strip().split("\t")
                except ValueError as e:
                    print(f"[ERROR] Line {lineno} malformed: {line.strip()}")
                    raise e

                qid2entry.setdefault(qid, {"query": query, "docs": []})
                qid2entry[qid]["docs"].append({"doc_id": pid, "text": passage})
        print(f"Loaded {len(qid2entry)} queries from {input_path}")
        return qid2entry
    except Exception as e:
        print("[FATAL] Exception occurred in load_msmarco_topn:")
        traceback.print_exc()
        raise e


def save_results(logger, results: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            # Write each result in the format: qid Q0 doc_id relevance system_name
            f.write(f"{item['query_id']} ignored {item['doc_id']} rank {item['score']} runid\n")
    logger.info(f"Reranking results saved to {output_path}.")


def setup_gpu_logging(worker_id, model_name_short):
    log_file = f"logs/rerank_worker{worker_id}_{model_name_short}.log"
    setup_logging(log_file)
    return logging.getLogger()


def rerank_worker(worker_id, qid_entry_chunk, args, generate_params, model_name_short):
    try:
        device_id = worker_id  # 让每个 worker 用 cuda:0, cuda:1, ...
        torch.cuda.set_device(device_id)
        logger = setup_gpu_logging(worker_id, model_name_short)

        reranker = PairwiseLlmRanker_HF(
            logger=logger,
            generate_params=generate_params,
            model_name_or_path=args.model_name,
            k=args.k,
            device=f"cuda:{device_id}"
        )

        results = []
        for qid, entry in qid_entry_chunk:
            query = entry["query"]
            docs = entry["docs"]
            reranked = reranker.rerank(qid, query, docs, bidirection=args.bidirection)
            results.extend(reranked)


        return {
            "results": results,
            "strict_match_count": reranker.strict_match_count,
            "approx_match_count": reranker.approx_match_count,
            "total_response_count": reranker.total_response_count
        }


    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error:")
        import traceback
        traceback.print_exc()
        raise e


def get_output_path(topn_path, model_name_short, query_num):
    topn_path = Path(topn_path)
    chunk_name = topn_path.stem  # e.g., chunk_02
    top_prefix = topn_path.parent.name  # e.g., top1000_english-german
    output_dir = Path(f"results/{top_prefix}/{model_name_short}")
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / f"{chunk_name}_{query_num}queries__rerank_results.tsv"


def cleanup():
    current_pid = os.getpid()
    for p in mp.active_children():
        print(f"Terminating subprocess {p.pid}")
        p.terminate()
    print(f"Process {current_pid} exiting.")
    sys.exit(0)


def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up...")
    cleanup()




def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--topn_path", type=str, default="data/top100.dev", help="Path to mMARCO topn file")
    parser.add_argument("--num_gpu", type=int, default=4, help="Number of GPU for evaluation.")
    # parser.add_argument("--qrels_path", type=str, default="data/qrels.dev.tsv", help="Path to MSMARCO qrels file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="LLM model name")
    parser.add_argument("--k", type=int, default=10, help="Number of top documents to rerank")
    # parser.add_argument("--top_n", type=int, default=100, help="Top N documents to load from top1000")
    parser.add_argument("--bidirection", action="store_true", help="Use bidirectional comparison")
    parser.add_argument("--query_num", type=int, default=None, help="Number of queries for evaluation.")

    args = parser.parse_args()

    args.model_name = os.path.abspath(args.model_name)

    path_parts = args.model_name.strip("/").split("/")
    model_name_short = "_".join(path_parts[-2:])
    log_file = f"logs/rerank_{model_name_short}.log"

    setup_logging(log_file)
    logger = logging.getLogger()

    topn_path = args.topn_path


    if os.path.exists(topn_path):
        logger.info(f"Found {topn_path}, loading from saved topN file.")
        qid2entry = load_msmarco_topn(topn_path)

    generate_params = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.85,
    }


    logger.info(f"Model {args.model_name} generation parameters: {generate_params}")

    if args.query_num == None:
        query_num = len(list(qid2entry.items()))
        output_path = get_output_path(args.topn_path, model_name_short, query_num)
    else:
        query_num = args.query_num
        output_path = get_output_path(args.topn_path, model_name_short, query_num)
    all_qid_entries = list(qid2entry.items())[:query_num]

    num_gpus = args.num_gpu

    chunk_size = math.ceil(len(all_qid_entries) / num_gpus)
    chunks = [all_qid_entries[i:i+chunk_size] for i in range(0, len(all_qid_entries), chunk_size)]

    worker_func = partial(rerank_worker,
                          args=args,
                          generate_params=generate_params,
                          model_name_short=model_name_short)

    with Pool(processes=num_gpus) as pool:
        results_dicts = pool.starmap(worker_func, enumerate(chunks))


    all_results = []
    total_strict = total_approx = total_resp = 0
    for r in results_dicts:
        all_results.extend(r["results"])
        total_strict += r["strict_match_count"]
        total_approx += r["approx_match_count"]
        total_resp += r["total_response_count"]

    strict_match_rate = total_strict / total_resp if total_resp else 0
    approx_match_rate = total_approx / total_resp if total_resp else 0


    logger.info(f"Strict match rate: {strict_match_rate} ({total_strict}/{total_resp})")
    logger.info(f"Approx match rate: {approx_match_rate} ({total_approx}/{total_resp})")

    save_results(logger, all_results, output_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        main()
    finally:
        cleanup()