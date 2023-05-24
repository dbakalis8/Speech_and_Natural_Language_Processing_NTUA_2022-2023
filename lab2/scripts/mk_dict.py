with open('lexicon.txt', 'r') as file:
    lines  = file.readlines()

distinct_phonemes = []

for line in lines:
    phonemes = line.split('\t')[1].strip().split()
    for phoneme in phonemes:
        if phoneme not in distinct_phonemes and phoneme != 'sil':
            distinct_phonemes.append(phoneme)

distinct_phonemes = sorted(distinct_phonemes)

with open('data/local/dict/nonsilence_phones.txt', 'w') as file:
    for phoneme in distinct_phonemes:
        file.write(phoneme + '\n')

with open('data/local/dict/lexicon.txt', 'w') as file:
    file.write('sil sil\n')
    for phoneme in distinct_phonemes:
        file.write(phoneme + ' ' + phoneme + '\n')

folders = ['train', 'test', 'dev']

for folder in folders:
    with open('data/' + folder + '/text') as file:
        lines  = file.readlines()

    with open('data/local/dict/lm_' + folder + '.text', 'w') as file:
        for line in lines:
            uttid, phonemes = line.split(' ', 1)
            file.write('<s> ' + phonemes[:-1] +  ' </s>\n')
