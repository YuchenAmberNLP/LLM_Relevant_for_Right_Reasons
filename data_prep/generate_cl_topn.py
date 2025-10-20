import random
import os
import argparse
import ftfy
from tqdm import tqdm

def clean_text(text: str) -> str:
    """Clean text using ftfy to fix encoding issues."""
    return ftfy.fix_text(text)

def load_lang_collection(lang_tsv_path):
    pid2de = {}
    with open(lang_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            pid, passage = parts
            pid2de[pid] = passage  # 不在这里处理，保持原始文本
    print(f"Loaded {len(pid2de)} entries from collection.")
    return pid2de

def load_lang_query(lang_tsv_path):
    qid2lang = {}
    with open(lang_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            qid, query = parts
            qid2lang[qid] = query  # 不在这里处理，保持原始文本
    print(f"Loaded {len(qid2lang)} entries from queries.")
    return qid2lang


def convert_cl_topn(topn_path, lang_query_path, lang_collection_path):
    topn_prefix = os.path.basename(topn_path).split(".")[0]
    pid2doc = load_lang_collection(lang_collection_path)
    qid2query = load_lang_query(lang_query_path)
    query_lang = os.path.basename(lang_query_path).rsplit('_', 1)[0]
    doc_lang = os.path.basename(lang_collection_path).rsplit('_', 1)[0]
    output_path = f"data/cl_topn/{topn_prefix}_{query_lang}-{doc_lang}.dev"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total = 0
    with open(topn_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue

            qid, pid, query, passage = parts
            if qid in qid2query:
                query = qid2query[qid]
            else:
                continue

            if pid in pid2doc:
                passage = pid2doc[pid]

            # Clean the final text before writing
            query = clean_text(query)
            passage = clean_text(passage)

            fout.write(f"{qid}\t{pid}\t{query}\t{passage}\n")
            total += 1
    print(f"Finished processing {total} lines.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MSMARCO top1000 passages to mixed multilingual passages")
    parser.add_argument("--topn_path", type=str, default="data/top100_shuffle.dev", help="Path to the input top1000 file")
    parser.add_argument("--lang_query_path", type=str, default="data/mmarco/queries/dev/english_queries.dev.small.tsv", help="Path to the queries.tsv file")
    parser.add_argument("--lang_collection_path", type=str, default="data/mmarco/collections/german_collection.tsv", help="Path to the collection.tsv file")

    args = parser.parse_args()

    convert_cl_topn(
        topn_path=args.topn_path,
        lang_query_path=args.lang_query_path,
        lang_collection_path=args.lang_collection_path
    )
