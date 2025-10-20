#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.

QUERIES_DIR="data/squad"
DOCS_DIR="data/squad"
QRELS_DIR="data/xquad-r"
OUTPUT_DIR="data/squad_train/"
NEG_SAMPLES=3

# download multilingual word embeddings
seen_languages=( ar en de it ru )
PREFIX=$( pwd )/muse_embeddings
mkdir -p $PREFIX
for LANG in "${seen_languages[@]}"
do
  wget "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.${LANG}.vec"
  mv "wiki.multi.${LANG}.vec" $PREFIX
done


mkdir -p "$OUTPUT_DIR"

while read -r qlang dlang; do
    [[ -z "${qlang:-}" || -z "${dlang:-}" ]] && continue
    echo "=== Generating JSONL for $qlang → $dlang ==="

    queries_file="$QUERIES_DIR/squad_${qlang}_queries.tsv"
    docs_file="$DOCS_DIR/squad_${dlang}_sentences.tsv"
    qrels_file="$QRELS_DIR/${dlang}_qrels.tsv"
    output_file="$OUTPUT_DIR/squad_${qlang}${dlang}_train.jsonl"

    if [[ ! -f "$queries_file" ]]; then
        echo "[Warning] Missing queries file: $queries_file"
        continue
    fi
    if [[ ! -f "$docs_file" ]]; then
        echo "[Warning] Missing docs file: $docs_file"
        continue
    fi
    if [[ ! -f "$qrels_file" ]]; then
        echo "[Warning] Missing qrels file: $qrels_file"
        continue
    fi

    python xquad_data/train_jsonl.py \
        --queries_file "$queries_file" \
        --docs_file "$docs_file" \
        --qrels_file "$qrels_file" \
        --output_file "$output_file" \
        --neg_samples "$NEG_SAMPLES"

done < data/xquad_langpair.txt


# generate code-switched training data
python xquad_data/code_switch.py \
    --input_file "data/squad_train/squad_enen_train.jsonl" \
    --path_embeddings "muse_embeddings" \
    --output_dir "data/squad_train"

# generate training data in instruction format
INS_OUTPUT_ROOT="instruction/squad"
mkdir -p "$INS_OUTPUT_ROOT"

find "$OUTPUT_DIR" -type f -name "train.jsonl" | while read -r input_path; do
    echo "=== Processing: $input_path ==="

    python generate_instruction_data.py \
        --input_path "$input_path" \
        --output_dir "$INS_OUTPUT_ROOT"
done