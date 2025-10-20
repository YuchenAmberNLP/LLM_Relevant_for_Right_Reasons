#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.


lang_full() {
  case "$1" in
    en) echo "english" ;;
    de) echo "german"  ;;
    ar) echo "arabic"  ;;
    it) echo "italian" ;;
    ru) echo "russian" ;;
    *)  echo "[ERROR] Unknown language code: $1" >&2; return 1 ;;
  esac
}

# download code-swithed and monolingual training data from public dataset from https://huggingface.co/datasets/rlitschk/csclir/tree/main
python data_prep/csdata_download.py

# generate mmarco instruction-tuning data
INPUT_ROOT="data/csclir"
OUTPUT_ROOT="instruction/mmarco"
mkdir -p "$OUTPUT_ROOT"

find "$INPUT_ROOT" -type f -name "train.jsonl" | while read input_path; do
    echo "=== Processing: $input_path ==="

    python generate_instruction_data.py \
        --input_path "$input_path" \
        --output_dir "$OUTPUT_ROOT"
done


