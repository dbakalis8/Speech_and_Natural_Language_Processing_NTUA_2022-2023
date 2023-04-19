from corpus_lexicon_construction import mk_transducer, mk_acceptor, mk_chars_words_indices
from util import CHARS

def mk_corpus_lexicon():
    with open('data/new_vocab.txt', 'r') as file:
        lines = file.readlines()


    with open('vocab/words.vocab.txt', "w") as file:
        for line in lines:
            flag = True
            word, freq = line.split()
            for letter in word:
                if letter not in CHARS:
                    flag = False
            if flag:
                file.write(word + '\t' + freq + '\n')

if __name__ == '__main__':
    mk_corpus_lexicon()
    mk_chars_words_indices()
    mk_transducer()
    mk_acceptor()
