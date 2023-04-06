import subprocess
import re

def mk_wiki_edits_file():
    with open('data/wiki.txt', 'r') as file:
        lines = file.readlines()

    with open('data/edits.txt', 'w') as file:
        for line in lines:
            line = re.sub(r'[^a-zA-Z\s]', '', line).split()
            misspelled_word = line[0]
            correct_word = line[1]
            subprocess.call(['./scripts/word_edits.sh', misspelled_word, correct_word])

if __name__ == '__main__':
    mk_wiki_edits_file()
