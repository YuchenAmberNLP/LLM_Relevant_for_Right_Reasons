import json
import random
import os
import argparse

def save_classification_triples(top1000_path: str, qrel_file: str, output_file: str, n_neg: int = 10):
    qid2positives = {}
    # 1. read all data in qrels
    with open(qrel_file, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, pid, label = line.strip().split()
            if label == "1":
                if qid not in qid2positives:
                    qid2positives[qid] = set()
                qid2positives[qid].add(pid)

    # read top1000 file
    all_entries = {}
    with open(top1000_path, "r", encoding="utf-8") as f:
        for line in f:
            qid, pid, query, passage = line.strip().split("\t")
            if qid not in all_entries:
                all_entries[qid] = []
            all_entries[qid].append(pid)

    # choose classification pair and save jsonl
    with open(output_file, "w", encoding="utf-8") as out_f:
        for qid, pid_list in all_entries.items():
            if qid not in qid2positives:
                continue
            posids = list(qid2positives[qid])
            negids = [pid for pid in pid_list if pid not in posids]
            if not negids:
                continue
            sampled_negs = random.sample(negids, min(n_neg, len(negids)))
            for posid in posids:
                entry = {
                    "qid": qid,
                    "posid": posid,
                    "negids": sampled_negs
                }
                out_f.write(json.dumps(entry) + "\n")


def load_lang_collection(lang_tsv_path):
    pid2lang = {}
    with open(lang_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            pid, passage = parts
            pid2lang[pid] = passage
    print(f"Loaded {len(pid2lang)} entries from collection.")
    return pid2lang


def load_lang_query(lang_tsv_path):
    qid2lang = {}
    with open(lang_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            qid, query = parts
            qid2lang[qid] = query
    print(f"Loaded {len(qid2lang)} entries from queries.")
    return qid2lang


def construct_classification_pairs(classification_path: str, query_file, collection_file, query_dir="data/mmarco/queries/dev/", collection_dir="data/mmarco/collections/", neg_ratio=None):

    query_path = os.path.join(query_dir, query_file)
    collection_path = os.path.join(collection_dir, collection_file)

    query_lang = os.path.basename(query_file).rsplit('_', 1)[0]
    collection_lang = os.path.basename(collection_file).rsplit('_', 1)[0]
    output_path = f"data/classification/{query_lang}-{collection_lang}_classification_pairs.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    qid2query = load_lang_query(query_path)
    pid2passage = load_lang_collection(collection_path)

    with open(classification_path, "r", encoding="utf-8") as f:
        classification_data = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in classification_data:
            qid = item["qid"]
            posid = item["posid"]
            negids = item["negids"]

            if qid not in qid2query or posid not in pid2passage:
                print("Query not in query file.")
                continue

            valid_negs = [pid for pid in negids if pid in pid2passage]
            if not valid_negs:
                continue
            query_text = qid2query[qid]
            pos_text = pid2passage[posid]
            if neg_ratio is None:
                sampled_negs = valid_negs
            else:
                sampled_negs = random.sample(valid_negs, min(neg_ratio, len(valid_negs)))

            for negid in sampled_negs:
                neg_text = pid2passage[negid]

                if random.random() < 0.5:
                    passage_A = {"pid": posid, "passage": pos_text}
                    passage_B = {"pid": negid, "passage": neg_text}
                    label = "Passage A"
                else:
                    passage_A = {"pid": negid, "passage": neg_text}
                    passage_B = {"pid": posid, "passage": pos_text}
                    label = "Passage B"

                output = {
                    "qid": qid,
                    "query": query_text,
                    "passage_A": passage_A,
                    "passage_B": passage_B,
                    "output": label
                }
                out_f.write(json.dumps(output) + "\n")

    print(f"Saved classification pairs to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MSMARCO top1000 passages to mixed multilingual passages")
    parser.add_argument("--topn_path", type=str, default="data/top1000.dev", help="Path to the input top1000 file")
    parser.add_argument("--lang_query_path", type=str, default="english_queries.dev.small.tsv", help="Path to the queries.tsv file")
    parser.add_argument("--lang_collection_path", type=str, default="multilingual_14l_collection.tsv", help="Path to the collection.tsv file")

    args = parser.parse_args()
    top1000_path = args.topn_path
    qrels_path = "data/qrels.dev.tsv"
    classification_ids_file = "data/classification_pair_ids.jsonl"
    save_classification_triples(top1000_path, qrels_path, classification_ids_file, n_neg=10)
    construct_classification_pairs(classification_ids_file, args.lang_query_path, args.lang_collection_path)
