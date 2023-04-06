#!/usr/bin/env bash
import sys
import subprocess

model = sys.argv[1]

with open('data/spell_test.txt', 'r') as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    if i==20:
        break
    line = line.split()
    correct_word = line[0].strip(':')
    misspelled_word = line[1]
    print('Misspelled word: ' + misspelled_word + ' | Correct word: ' + correct_word + ' | Predicted word: ', end='', flush=True)
    subprocess.call(['./scripts/predict.sh', './fsts/' + model + '.binfst', misspelled_word])
    print()
