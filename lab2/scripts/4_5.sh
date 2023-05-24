#!/usr/bin/env bash
source path.sh

for set in train dev test; do
  bash steps/align_si.sh data/$set data/lang exp/tri_bg exp/tri_align_$set
done

bash scripts/run_dnn.sh
