import torch

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tweets, targets):
        """
        tweets: numpy array
        targets: a vector, numpy array
        """
        self.tweets = tweets
        self.targets = targets

    def __len__(self):
        # return the length of the dataset
        return len(self.tweets)
    
    def __getitem__(self, item):
        # for any given item, which is an int
        # return tweets and targets as torch tensor
        tweet = self.tweets[item, :]
        target = self.targets[item]
        return {
            "tweet": torch.tensor(tweet, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }


