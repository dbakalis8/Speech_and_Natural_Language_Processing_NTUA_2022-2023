source path.sh

# bash steps/train_mono.sh data/train data/lang exp/mono
#
# for n in ug bg; do
#   cp -R exp/mono exp/mono_$n
#   bash scripts/mkgraph_dec_eval.sh mono $n
# done
#
# bash steps/align_si.sh data/train data/lang exp/mono exp/mono_align
#
# bash steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_align exp/tri
#
# for n in ug bg; do
#   cp -R exp/tri exp/tri_$n
#   bash scripts/mkgraph_dec_eval.sh tri $n
# done
#
# bash steps/align_si.sh data/train data/lang exp/tri exp/tri_align


for model in mono tri; do
  if [ $model == 'mono' ]; then
    bash steps/train_mono.sh data/train data/lang exp/mono
  else
    bash steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_align exp/tri
  fi

  for n in ug bg; do
    cp -R exp/$model exp/${model}_${n}
    bash scripts/mkgraph_dec_eval.sh ${model} ${n}
  done

  bash steps/align_si.sh data/train data/lang exp/$model exp/${model}_align
done

for model in mono tri; do
  bash scripts/print_results.sh $model
done
