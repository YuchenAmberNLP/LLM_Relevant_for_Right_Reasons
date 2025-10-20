#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.


# download mmarco multilingual collections (documents) and queries 
python data_prep/mmarco_download.py



# generate top100.dev
python data_prep/generate_top100_qrels.py

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


TOPN_PATH="data/top100_shuffle.dev"
QUERY_DIR="data/mmarco/queries/dev"
COLLECTION_DIR="data/mmarco/collections"

while read -r qlang clang; do
  [[ -z "${qlang:-}" || -z "${clang:-}" ]] && continue
  echo "=== Running for $qlang - $clang ==="

  qfull=$(lang_full "$qlang")
  cfull=$(lang_full "$clang")

  query_path="$QUERY_DIR/${qfull}_queries.dev.tsv"
  collection_path="$COLLECTION_DIR/${cfull}_collection.tsv"

  if [[ ! -f "$query_path" ]]; then
    echo "[Warning] Query file not found: $query_path"
    continue
  fi
  if [[ ! -f "$collection_path" ]]; then
    echo "[Warning] Collection file not found: $collection_path"
    continue
  fi

  python data_prep/generate_cl_topn.py \
    --topn_path "$TOPN_PATH" \
    --lang_query_path "$query_path" \
    --lang_collection_path "$collection_path"
done < data/mmarco_langpairs.txt


