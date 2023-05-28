import torch

from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
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
        self.nonlinear_trans = nn.ReLU()  # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.output_layer = nn.Linear(50, output_size) # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.emb_layer(x)  # EX6

        # 2 - construct a sentence representation out of the word embeddings

        representations = torch.stack([torch.mean(torch.t(sent[:lengths[i]]), 1) for i, sent in enumerate(embeddings)])  # EX6

        # 3 - transform the representations to new ones.
        representations = self.nonlinear_trans(representations)  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.output_layer(representations)  # EX6

        return logits
