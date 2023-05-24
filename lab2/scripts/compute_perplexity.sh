source path.sh

compile-lm data/local/lm_tmp/uni_train.ilm.gz -eval=data/local/dict/lm_dev.text --dub=10000000
compile-lm data/local/lm_tmp/uni_train.ilm.gz -eval=data/local/dict/lm_test.text --dub=10000000

compile-lm data/local/lm_tmp/bi_train.ilm.gz -eval=data/local/dict/lm_dev.text --dub=10000000
compile-lm data/local/lm_tmp/bi_train.ilm.gz -eval=data/local/dict/lm_test.text --dub=10000000
