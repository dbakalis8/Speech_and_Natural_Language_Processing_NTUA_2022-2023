import re
import string
folders = ['train', 'dev', 'test']

for folder in folders:
    with open('data/' + folder + '/uttids', 'r') as file:
        uttids = file.readlines()

    with open('transcriptions.txt', 'r') as file:
        transcriptions = file.readlines()

    with open('lexicon.txt', 'r') as file:
        lexicon = file.readlines()

    lexicon = {line.split('\t', 1)[0].lower(): line.split('\t', 1)[1][:-1] for line in lexicon}

    with open('data/' + folder + '/text', 'w') as file:
        for uttid in uttids:
            file_line = int(uttid[3:]) - 1
            sentence = transcriptions[file_line][4:].replace("-", " ")
            punctuation = string.punctuation.replace("'", "")
            sentence = re.sub(r"[{}]".format(punctuation), "", sentence).lower()

            phonemes = 'sil'
            for word in sentence.split():
                phonemes += lexicon[word]
            phonemes += ' sil\n'

            file.write(uttid[:-1] + ' ' + phonemes)
