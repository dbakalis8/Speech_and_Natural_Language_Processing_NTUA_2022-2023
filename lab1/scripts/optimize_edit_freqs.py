from collections import Counter
from util import EPS, CHARS, calculate_arc_weight, format_arc

V = 702
def edit_freqs_with_add1_smoothing():
    with open('data/edits.txt', 'r') as file:
        lines = file.readlines()

    edits = []
    for line in lines:
        line = line.split()
        edits.append((line[0], line[1]))

    edit_freq = Counter(edits)

    chars_list = [EPS]
    chars_list.extend(CHARS)
    chars_dict = {char:0 for char in chars_list}

    for (source, target), freq in edit_freq.items():
        chars_dict[source] += freq

    edit_prob = {(source, target): (freq+1)/(chars_dict[source] + V) for (source, target), freq in edit_freq.items()}

    for char1 in chars_list:
        for char2 in chars_list:
            if char1 != char2:
                if (char1, char2) in edit_freq:
                    continue
                else:
                    edit_prob[(char1, char2)] = 1/(chars_dict[char1] + V)
    return edit_prob

def mk_optimized_E(edit_prob):
    with open('vocab/chars.syms', 'r') as file:
        lines = file.readlines()

    chars = []
    for line in lines:
        chars.append(line.split('\t')[0])

    #chars = [EPS, 'a', 'b', 'c', 'd']           #used to simplify the drawing of the transducer

    with open('fsts/E_opt.fst', 'w') as file:
        for char in chars:
            file.write(format_arc(0, 0, char, char))                        #o edit

        for char in chars[1:]:
            freq = edit_prob[(char, EPS)]
            weight = calculate_arc_weight(freq)
            file.write(format_arc(0, 0, char, EPS, weight))                 #deletion

        for char in chars[1:]:
            try:
                freq = edit_prob[(EPS, char)]
            except:
                freq = 0
            weight = calculate_arc_weight(freq)
            file.write(format_arc(0, 0, EPS, char, weight))                 #insertion

        for char1 in chars[1:]:
            for char2 in chars[1:]:
                if char1 != char2:
                    try:
                        freq = edit_prob[(char1, char2)]
                    except:
                        freq = 0
                    weight = calculate_arc_weight(freq)
                    file.write(format_arc(0, 0, char1, char2, weight))      #substitution

        file.write("0")

if __name__ == '__main__':
    edit_probs = edit_freqs_with_add1_smoothing()
    mk_optimized_E(edit_probs)
