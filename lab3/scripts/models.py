import torch

from torch import nn
import numpy as np


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False, max_pooling=True):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        self.num_embeddings, self.embeddings_dim = embeddings.shape
        self.emb_layer = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embeddings_dim)  # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.emb_layer.weight = nn.Parameter(torch.from_numpy(embeddings)) # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        self.emb_layer.weight.requires_grad = trainable_emb # EX4

        # 4 - define a non-linear transformation of the representations
        self.max_pooling = max_pooling
        self.linear = nn.Linear(2*self.embeddings_dim, 1000) if self.max_pooling else nn.Linear(self.embeddings_dim, 1000)
        self.nonlinear_trans = nn.ReLU()  # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.output_layer = nn.Linear(1000, output_size) # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.emb_layer(x)  # EX6

        # 2 - construct a sentence representation out of the word embeddings

        avg_pooling = torch.stack([torch.mean(torch.t(sent[:lengths[i]]), 1) for i, sent in enumerate(embeddings)])  # EX6
        max_pooling = torch.stack([torch.max(torch.t(sent[:lengths[i]]), 1)[0] for i, sent in enumerate(embeddings)])
        representations = torch.cat([avg_pooling, max_pooling], dim=1) if self.max_pooling else avg_pooling

        # 3 - transform the representations to new ones.

        representations = self.nonlinear_trans(self.linear(representations))  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.output_layer(representations)  # EX6

        return logits
    
class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        
        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)

        representations = torch.zeros(batch_size, self.representation_size).float()
        for i in range(lengths.shape[0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]

        logits = self.linear(representations)

        return logits
