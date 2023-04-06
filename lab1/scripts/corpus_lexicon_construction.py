from collections import Counter
from util import format_arc, EPS, CHARS
import subprocess

FINAL_STATE = 100000000

def make_corpus_lexicon():
    with open('data/gutenberg.txt', 'r') as file:
        corpus = file.readlines()

    words = []
    for line in corpus:
        words.extend(line[:-1].split())

    dict_of_words = Counter(words)
    threshold = 5
    filtered_dict = {word: freq for word, freq in dict_of_words.items() if freq >= threshold}

    with open('vocab/words.vocab.txt', "w") as file:
        for word, freq in filtered_dict.items():
            file.write(word + '\t' + str(freq) + '\n')

    return True

def mk_chars_words_indices():
    chars = [EPS]
    chars.extend(CHARS)

    with open('vocab/chars.syms', "w") as file:
        for i, char in enumerate(chars):
            file.write(char + '\t' + str(i) + '\n')

    with open('vocab/words.vocab.txt', "r") as file:
        lines = file.readlines()

    words = [EPS]
    for line in lines:
        words.append(line.split('\t')[0])

    with open('vocab/words.syms', "w") as file:
        for i, word in enumerate(words):
            file.write(word + '\t' + str(i) + '\n')

    return True

def mk_transducer():
    with open('vocab/chars.syms', 'r') as file:
        lines = file.readlines()

    chars = []
    for line in lines:
        chars.append(line.split('\t')[0])

    #chars = [EPS, 'a', 'b', 'c', 'd']           #used to simplify the drawing of the transducer

    with open('fsts/L.fst', 'w') as file:
        for char in chars:
            file.write(format_arc(0, 0, char, char))                    #no edit

        for char in chars[1:]:
            file.write(format_arc(0, 0, char, EPS, 1))                  #deletion

        for char in chars[1:]:
            file.write(format_arc(0, 0, EPS, char, 1))                  #insertion

        for char1 in chars[1:]:
            for char2 in chars[1:]:
                if char1 != char2:
                    file.write(format_arc(0, 0, char1, char2, 1))       #replacement

        file.write("0")

def mk_acceptor():
    with open('vocab/words.syms', 'r') as file:
        lines = file.readlines()

    words = []
    for line in lines:
        words.append(line.split('\t')[0])

    #words = words[:10]                           #used to simplify the drawing of the acceptor

    with open('fsts/V.fst', 'w') as file:
        #file.write(format_arc(0, 1, EPS, EPS))
        #file.write("1\n")
        cnt = 1
        for word in words[1:]:
            for j, letter in enumerate(word):
                if j == 0:
                    file.write(format_arc(0, cnt, letter, word))
                else:
                    file.write(format_arc(cnt, cnt+1, letter, EPS))
                    cnt += 1
            if word == words[-1]:
                file.write(str(cnt))
            else:
                file.write(str(cnt) + '\n')
            cnt += 1



if __name__ == "__main__":
    make_corpus_lexicon()
    mk_chars_words_indices()
    mk_transducer()
    mk_acceptor()
