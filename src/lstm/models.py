import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        """
        :param embedding_matrix: numpy array with vectors for all words
        """
        super(LSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        # input embedding layer
        self.embedding = nn.Embedding(num_words, embed_dim)
        # embedding_matrix is used as weights for the embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
                )
            )
        # don't train the pretrained embeddings
        self.embedding.weight.requires_grad = False
        # bidirectional LSTM layer with 128 hidden units
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True
        )
        # use a linear layer to get the final output
        # 512 = 128 + 128 for mean and max pooling on both directions
        self.out = nn.Linear(512, 1)
        
    def forward(self, x):
        # pass the input through the embedding layer
        x = self.embedding(x)
        # pass the embeddings to the LSTM layer
        x, _ = self.lstm(x)
        # apply mean and max pooling on the LSTM output
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        out = torch.cat((avg_pool, max_pool), 1)
        # pass through the output layer and return
        out = self.out(out)
        return out