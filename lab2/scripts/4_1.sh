ln -s ../wsj/s5/steps steps
ln -s ../wsj/s5/utils utils

mkdir local

ln -s ../steps/score_kaldi.sh local/score.sh

mkdir conf

cp ../wsj/s5/conf/mfcc.conf ./conf
echo --sample-frequency=22050 >> conf/mfcc.conf

mkdir data/lang data/local data/local/dict data/local/lm_tmp data/local/nist_lm
