# dataset.py
import config
import torch

class ROBERTADataset:
    def __init__(self, tweets, labels, lengths):
        """
        :param tweets: list or numpy arrays of strings
        :param labels: list or numpy arrays of binary numbers
        :param lengths: list or numpy arrays of numbers
        """
        self.tweets = tweets
        self.labels = labels
        self.lengths = lengths
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)