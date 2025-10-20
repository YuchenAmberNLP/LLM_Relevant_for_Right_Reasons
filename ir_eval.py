import pandas as pd
import ir_measures
from ir_measures import *
import logging
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--qrels_path", type=str, default="data/qrels.dev.tsv", help="Path to qrels file")
parser.add_argument("--run_path", type=str, required=True, help="Path to result file")

args = parser.parse_args()

qrel_path = args.qrels_path
run_path = args.run_path
tmp_qrel_path = 'data/tmp_qrels.tsv'

run_qids = set()
with open(run_path, 'r') as run_f:
    for line in run_f:
        qid = line.split()[0]
        run_qids.add(qid)

with open(qrel_path, 'r') as qrels_in, open(tmp_qrel_path, 'w') as qrels_out:
    for line in qrels_in:
        qid = line.split()[0]
        if qid in run_qids:
            qrels_out.write(line)


qrels = ir_measures.read_trec_qrels(tmp_qrel_path)
run = ir_measures.read_trec_run(run_path)
metrics = ir_measures.calc_aggregate([RR@10], qrels, run)

print(f"Results of {args.run_path}:")
print(metrics)
logging.info(f"Results of {args.run_path}:")
logging.info(metrics)

os.remove(tmp_qrel_path)