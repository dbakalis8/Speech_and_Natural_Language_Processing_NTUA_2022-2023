source path.sh

model=$1
ngram=$2

bash utils/mkgraph.sh data/lang_phones_$ngram exp/${model}_${ngram} exp/${model}_${ngram}/graph
for folder in dev test; do
  bash steps/decode.sh exp/${model}_${ngram}/graph data/$folder exp/${model}_${ngram}/decode_$folder
  bash local/score.sh --cmd "run.pl" data/$folder exp/${model}_${ngram}/graph exp/${model}_${ngram}/decode_$folder
done
