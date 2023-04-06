import logging
import multiprocessing
import os
import nltk
nltk.download('punkt')

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# Enable gensim logging
logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


class W2VLossLogger(CallbackAny2Vec):
    """Callback to print loss after each epoch
    use by passing model.train(..., callbacks=[W2VLossLogger()])
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()

        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print(
                "Loss after epoch {}: {}".format(
                    self.epoch, loss - self.loss_previous_step
                )
            )
        self.epoch += 1
        self.loss_previous_step = loss


def train_w2v_model(
    sentences,
    output_file,
    window=5,
    embedding_dim=100,
    epochs=1000,
    min_word_count=10,
):
    """Train a word2vec model based on given sentences.
    Args:
        sentences list[list[str]]: List of sentences. Each element contains a list with the words
            in the current sentence
        output_file (str): Path to save the trained w2v model
        window (int): w2v context size
        embedding_dim (int): w2v vector dimension
        epochs (int): How many epochs should the training run
        min_word_count (int): Ignore words that appear less than min_word_count times
    """

    workers = multiprocessing.cpu_count()

    # TODO: Instantiate gensim.models.Word2Vec class
    model = Word2Vec(sentences=sentences, size=embedding_dim, window=window, min_count=min_word_count, workers=workers)
    # TODO: Build model vocabulary using sentences
    # TODO: Train word2vec model
    model.train(sentences, total_examples=len(sentences), epochs=1000, callbacks=[W2VLossLogger()])
    # Save trained model
    model.save(output_file)

    return model


if __name__ == "__main__":
    # read data/gutenberg.txt in the expected format
    # nltk.corpus.gutenberg.sents is used in order to read sentences not lines.
    sentences = nltk.corpus.gutenberg.sents('austen-emma.txt')
    output_file = "gutenberg_w2v.100d.model"
    window = 5
    embedding_dim = 100
    epochs = 1000
    min_word_count = 10

    train_w2v_model(
        sentences,
        output_file,
        window=window,
        embedding_dim=embedding_dim,
        epochs=epochs,
        min_word_count=min_word_count,
    )
