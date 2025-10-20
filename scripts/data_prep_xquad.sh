#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.

# generate xquad top100 data
N=100
while read -r qlang slang; do
  [[ -z "${qlang:-}" || -z "${slang:-}" ]] && continue
  echo "=== Generating XQuAD top-$N for $qlang → $slang ==="

  python xquad_data/xquad_topn_gen.py \
    --query_lang "$qlang" \
    --sent_lang "$slang" \
    --n "$N"
done < data/xquad_langpair.txt

