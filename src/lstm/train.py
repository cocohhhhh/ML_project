import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from helpers import *

if __name__ == '__main__':
    # load the data
    with open('../data/twitter-datasets/train_pos.txt', 'r') as f:
        pos_tweets = f.readlines()
    with open('../data/twitter-datasets/train_neg.txt', 'r') as f:
        neg_tweets = f.readlines()

    # create the dataframe
    pos_tweets = pd.DataFrame(pos_tweets, columns=['tweet'])
    pos_tweets['target'] = 1
    neg_tweets = pd.DataFrame(neg_tweets, columns=['tweet'])
    neg_tweets['target'] = 0
    df = pd.concat([pos_tweets, neg_tweets], ignore_index=True)

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    run_lstm(df, 5)