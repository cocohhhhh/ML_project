import numpy as np
import pandas as pd
import torch

from helpers import *
import config

if __name__ == '__main__':
    # load the data
    data = []

    with open('../data/twitter-datasets/test_data.txt', 'r') as file:
        for line in file:
            # Split the line at the first comma
            id_str, tweet = line.strip().split(',', 1)
            data.append((id_str, tweet))

    # create the dataframe
    df = pd.DataFrame(data, columns=['Id', 'tweet'])

    # load the model
    model = torch.load('../manipulated/model.pkl')

    # fitting tokenizer on training data
    x_test = df['tweet'].values
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x_test)
    x_test = tokenizer.texts_to_sequences(x_test)

    # pad the sequences with zeros
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=config.MAX_LEN)
    predictions = predict(x_test, model, 'cuda')

    # save the predictions
    df['Prediction'] = predictions
    df = df[['Id', 'Prediction']]

    # save the dataframe
    df.to_csv('../manipulated/predictions/lstm_predictions.csv', index=False)



