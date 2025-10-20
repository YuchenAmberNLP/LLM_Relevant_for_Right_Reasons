import json
import random
top1000_path = "data/top1000.dev"
qrels_path = "data/qrels.dev.tsv"

valid_qids = set()
top1000_qrels_set = set()

# generate top100.dev file
def load_msmarco_top1000(path: str, qrel_file=None, top_n: int = 1000, small_path=None):

    qid2entry = {}
    qid2positives = {}

    # load positive samples
    if qrel_file is not None and top_n < 1000:
        with open(qrel_file, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, pid, label = line.strip().split()
                if label == "1":
                    if qid not in qid2positives:
                        qid2positives[qid] = set()
                    qid2positives[qid].add(pid)

    all_entries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, pid, query, passage = line.strip().split("\t")
            if qid not in all_entries:
                all_entries[qid] = {
                    "query": query,
                    "docs": []
                }
            all_entries[qid]["docs"].append({
                "doc_id": pid,
                "text": passage
            })

    for qid, data in all_entries.items():
        query = data["query"]
        docs = data["docs"]

        if qid not in qid2positives:
            print(f"Qid {qid} doesn't have positive docs in top 1000 file.")

        else:
            positive_pids = qid2positives[qid]
            positive_docs = [doc for doc in docs if doc["doc_id"] in positive_pids]
            other_docs = [doc for doc in docs if doc["doc_id"] not in positive_pids]

            if not positive_docs:
                print("No positive_docs in top 1000 file")
                # selected_docs = docs[:top_n]
            else:
                remaining = top_n - len(positive_docs)
                sampled = random.sample(other_docs, remaining) if remaining <= len(other_docs) else other_docs
                selected_docs = positive_docs + sampled
                random.shuffle(selected_docs)

                qid2entry[qid] = {
                    "query": query,
                    "docs": selected_docs
                }

    if small_path:
        with open(small_path, "w", encoding="utf-8") as fout:
            for qid, entry in qid2entry.items():
                query = entry["query"]
                for doc in entry["docs"]:
                    fout.write(f"{qid}\t{doc['doc_id']}\t{query}\t{doc['text']}\n")

        print(f"Saved {len(qid2entry)} entries to {small_path}")
    return qid2entry


load_msmarco_top1000(top1000_path, qrels_path, 100, "data/top100_shuffle.dev")