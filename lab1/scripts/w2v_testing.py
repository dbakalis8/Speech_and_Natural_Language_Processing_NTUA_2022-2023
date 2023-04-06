from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec, KeyedVectors
import sys

NUM_W2V_TO_LOAD = 1000000

#cosine_distance is taken from https://github.com/slp-ntua/slp-labs/blob/master/lab1/examples/gensim/Genism%20Word2Vec%20Tutorial.ipynb
def cosine_distance (model, word, target_list, num):
    cosine_dict = {}
    word_list = []
    a = model[word]
    for item in target_list:
        if item != word:
            b = model[item]
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

if __name__ == '__main__':
    if sys.argv[1] == '0':
        model = Word2Vec.load("gutenberg_w2v.100d.model")
    else:
        model = KeyedVectors.load_word2vec_format('GoogleNewsvectorsnegative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)

    w2v = model.wv
    test_words = ['bible', 'book', 'bank', 'water']

    with open('vocab.syms', 'r') as file:
        lines = file.readlines()

    target_list = [line.split()[0] for line in lines]

    for word in test_words:
        print('Most similar words to ' + word + ':')
        sim_words = cosine_distance(w2v, word, target_list, 5)
        for i, sim_word in enumerate(sim_words):
            print(str(i) + ': ' + sim_word[0] + ' with cosine similarity ' + sim_word[1])
        print('\n' + 50*'=' + '\n')

    test_triplets = [('girl', 'queen', 'king'), ('taller', 'tall', 'good'), ('france', 'paris', 'london')]

    for triplet in test_triplets:
        v = w2v[tiplet[0]] - w2v[triplet[1]] + w2v[triplet[2]]
        most_sim = cosine_distance(w2v, v, target_list, 1)[0]
        print('Given the words', end = ' ')
        for word in triplet:
            print(word, end = ', ')
        print('the most similar word is ' + most_sim[0] + ' with cosine similarity ' + most_sim[1])
        print('\n' + 50*'=' + '\n')
