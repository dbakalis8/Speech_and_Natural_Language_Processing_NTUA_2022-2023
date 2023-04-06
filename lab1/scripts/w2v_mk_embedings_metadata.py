from gensim.models import Word2Vec

model = Word2Vec.load("gutenberg_w2v.100d.model")
w2v = model.wv

with open('vocab/words.syms') as file:
    lines = file.readlines()

words = [line.split()[0] for line in lines)]

with open('data/embeddings.tsv') as file:
    for word in words:
        embedding = w2v[word]
        for i, num in enumerate(embedding):
            if i == 99:
                file.write(num + '\n')
            else:
                file.write(num + '\t')

with open('data/metadata.tsv') as file:
    for word in words:
        file.write(word + '\n')
