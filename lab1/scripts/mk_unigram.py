from util import EPS, CHARS, format_arc, calculate_arc_weight

def mk_unigram():

    with open('vocab/words.vocab.txt', 'r') as file:
        lines = file.readlines()

    words = []
    total_words = 0
    for line in lines:
        line = line.split()
        words.append(line)
        total_words += int(line[1])

    word_freqs = {word: int(freq)/total_words for word, freq in words}

    #words = words[:4]                          #used to simplify the drawing of the acceptor

    with open('fsts/W.fst', 'w') as file:
        weight = calculate_arc_weight(0)
        file.write(format_arc(0, 0, EPS, EPS, weight))

        for word in words:
            word = word[0]
            try:
                freq = word_freqs[word]
            except:
                freq = 0

            weight = calculate_arc_weight(freq)
            file.write(format_arc(0, 0, word, word, weight))

        file.write("0")

if __name__ == '__main__':
    mk_unigram()
