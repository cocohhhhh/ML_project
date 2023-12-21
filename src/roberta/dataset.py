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
        """
        for a given index, return a dictionary of tensors
        """
        tweet = str(self.tweets[idx])
        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        item = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.labels[idx], dtype=torch.float),
            "lengths": torch.tensor(self.lengths[idx], dtype=torch.float)
        }
        return item

    def __len__(self):
        return len(self.labels)
    

class ROBERTATestDataset:
    def __init__(self, tweets, lengths):
        """
        :param tweets: list or numpy arrays of strings
        :param lengths: list or numpy arrays of numbers
        """
        self.tweets = tweets
        self.lengths = lengths
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __getitem__(self, idx):
        """
        for a given index, return a dictionary of tensors
        """
        tweet = str(self.tweets[idx])
        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        item = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "lengths": torch.tensor(self.lengths[idx], dtype=torch.float)
        }
        return item

    def __len__(self):
        return len(self.tweets)