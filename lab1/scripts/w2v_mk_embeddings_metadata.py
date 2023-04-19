from gensim.models import Word2Vec
from w2v_train import W2VLossLogger

w2v = Word2Vec.load("gutenberg_w2v.100d.model").wv

with open('vocab/words.syms', 'r') as file:
    lines = file.readlines()

words = [line.split()[0] for line in lines]

with open('data/embeddings.tsv', 'w') as file:
    not_in_vocab = []
    for word in words:
        try:
            embedding = w2v[word]
        except:
            not_in_vocab.append(word)
            continue
        for i, num in enumerate(embedding):
            if i == w2v.vector_size-1:
                file.write(str(num) + '\n')
            else:
                file.write(str(num) + '\t')

with open('data/metadata.tsv', 'w') as file:
    for word in words:
        if word in not_in_vocab:
            continue
        else:
            file.write(word + '\n')
