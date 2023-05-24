source path.sh

touch data/local/dict/silence_phones.txt
touch data/local/dict/optional_silence.txt

echo 'sil' > data/local/dict/silence_phones.txt
echo 'sil' > data/local/dict/optional_silence.txt

python scripts/mk_dict.py

touch data/local/dict/extra_questions.txt

build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/uni_train.ilm.gz
build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/bi_train.ilm.gz

compile-lm data/local/lm_tmp/uni_train.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_ug.arpa.gz
compile-lm data/local/lm_tmp/bi_train.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_bg.arpa.gz

prepare_lang.sh data/local/dict "<oov>" data/local/tmp data/lang

for folder in train test dev; do
  for file in wav.scp text utt2spk; do
    sort data/${folder}/${file} > tmp
    cat tmp > data/$folder/$file
    if [ $file == 'utt2spk' ]
    then
      utt2spk_to_spk2utt.pl data/${folder}/${file} > data/${folder}/spk2utt
    fi
  done
done

rm tmp

bash scripts/timit_format_data.sh
