import torch
from torch import nn


class RNNTextClassifier(nn.Module):
    """A text classifier based on RNN.

    :param vocab_len: Length of the vocab.
    :param class_num: Number of labels.
    :param embed_dim: Embedding dimension.
    :param hidden_dim: Dimension of the RNN hidden layers.
    """

    def __init__(self, vocab_len: int, class_num: int, embed_dim: int, hidden_dim: int):
        super(RNNTextClassifier, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_len,
            embedding_dim=self.embed_dim
        )
        self.rnn = nn.RNN(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=self.hidden_dim,
            out_features=class_num
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        embeddings = self.embedding(x)

        _, last_hidden = self.rnn(embeddings)
        last_hidden = last_hidden.squeeze(dim=0)

        y = self.linear(last_hidden)

        return y
