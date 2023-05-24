source path.sh

for folder in train dev test; do
  bash steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --cmd "run.pl" data/$folder exp/mk_mfcc/$folder mfcc_$folder
  bash steps/compute_cmvn_stats.sh data/$folder exp/make_mfcc/$train mfcc
done
