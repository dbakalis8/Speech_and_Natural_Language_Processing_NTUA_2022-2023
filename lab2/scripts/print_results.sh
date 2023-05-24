source path.sh

model=$1

echo Printing results for $model phone...

for n in ug bg; do
  for folder in dev test; do
    echo $folder set, $n:
    cat exp/${model}_${n}/decode_${folder}/scoring_kaldi/best_wer
  done
done
