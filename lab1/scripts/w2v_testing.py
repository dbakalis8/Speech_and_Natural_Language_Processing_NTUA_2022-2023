from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec, KeyedVectors
import sys
from w2v_train import W2VLossLogger

NUM_W2V_TO_LOAD = 1000000

#cosine distance taken from https://github.com/slp-ntua/slp-labs/blob/master/lab1/examples/gensim/Genism%20Word2Vec%20Tutorial.ipynb
def cosine_distance (model, word, target_list , num, flag):
    cosine_dict = {}
    word_list = []
    if flag:                                     #if flag is true we receive a word, else we receive a vector.
        a = model[word]

    else:
        a = word
        word = None

    for item in target_list:
        if item != word :
            try:
                b = model[item]
            except:
                continue
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            cosine_dict[item] = cos_sim
    dist_sort = sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descending order
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

if __name__ == '__main__':
    if sys.argv[1] == '0':
        w2v = Word2Vec.load('gutenberg_w2v.100d.model').wv
    else:
        w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)

    test_words = ['bible', 'book', 'bank', 'water']

    with open('vocab/words.syms', 'r') as file:
        lines = file.readlines()

    target_list = [line.split()[0] for line in lines]

    for word in test_words:
        print('Most similar words to ' + word + ':')
        sim_words = cosine_distance(w2v, word, target_list, 5, flag=True)
        for i, sim_word in enumerate(sim_words):
            print(str(i) + ': ' + sim_word[0] + ' with cosine similarity ' + str(sim_word[1]))
        print('\n' + 100*'=' + '\n')

    test_triplets = [('girl', 'queen', 'king'), ('taller', 'tall', 'good'), ('france', 'paris', 'london')]

    for triplet in test_triplets:
        v = w2v[triplet[0]] - w2v[triplet[1]] + w2v[triplet[2]]
        #print(cosine_distance(w2v, v, target_list, 5, flag=False))
        most_sim = cosine_distance(w2v, v, target_list, 5, flag=False)
        print('Given words', end=' ')
        for word in triplet:
            print(word, end=', ')
        words = [word for (word, sim) in most_sim]
        sims = [sim for (word, sim) in most_sim]
        print('the most similar words are ' + str(words) + ' with cosine similarity ' + str(sims))
        print('\n' + 100*'=' + '\n')
