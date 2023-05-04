#!/usr/bin/env bash

mkdir data
mkdir data/train data/test data/dev

cp filesets/training.txt data/train
mv data/train/training.txt data/train/uttids

cp filesets/testing.txt data/test
mv data/test/testing.txt data/test/uttids

cp filesets/validation.txt data/dev
mv data/dev/validation.txt data/dev/uttids

python scripts/mk_utt2spk.py
python scripts/mk_wavs.py
python scripts/mk_text.py

echo 'All files created'
