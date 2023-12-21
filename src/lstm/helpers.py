import io
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn

import tensorflow as tf

from sklearn import metrics
from sklearn import model_selection

import config
import dataset
import models

def train(data_loader, model, optimizer, device):
    """
    Main training function for one epoch
    :param data_loader: torch data loader
    :param model: torch model(lstm,cnn, etc.)
    :param optimizer: optimizer, e.g. Adam, sgd, etc.
    :param device: cuda or cpu
    """
    # set model to training mode
    model.train()
    # go through all batches in data loader
    for data in data_loader:
        tweets = data["tweet"]
        targets = data["target"]
        tweets = tweets.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # clear the gradients
        optimizer.zero_grad()

        # make predictions from the model
        predictions = model(tweets)

        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(
            predictions,
            targets.view(-1, 1)
        )

        # compute gradients of loss w.r.t. all parameters of the model
        loss.backward()

        # single optimization step
        optimizer.step()


def evaluate(data_loader, model, device):
    """
    Function for evaluating the model
    :param data_loader: torch data loader
    :param model: torch model(lstm, cnn, etc.)
    :param device: cuda or cpu
    """
    final_predictions = []
    final_targets = []

    # put the model in eval mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            tweets = data["tweet"]
            targets = data["target"]
            tweets = tweets.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # make predictions
            predictions = model(tweets)

            # move predictions and targets to list
            predictions = predictions.cpu().numpy().tolist()
            targets = data["target"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)

    # return final predictions and targets
    return final_predictions, final_targets

def load_vectors(fname):
    """
    Load the pretrained vectors from a file
    :param fname: path to the pretrained vectors file
    """
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    fin.close()
    return data

def create_embedding_matrix(word_index, embedding_dict):
    """
    Create the embedding matrix from the embedding dictionary
    :param word_index: a dictionary with word:index_value
    :param embedding_dict: a dictionary with word:embedding_vector
    :return: a numpy array with embedding vectors for all known words
    """
    # initialize the embedding matrix as a numpy array of zeros
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    # loop over all the words
    for word, i in word_index.items():
        # if word is found in pre-trained embeddings, update the matrix
        # at index i with its embedding
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix

def run_lstm(df, folds):
    """
    Run training and validation for a given fold
    :param df: pandas dataframe with tweet and target columns
    :param folds: total number of folds, int
    """
    
    # Initialize KFold from scikit-learn
    kf = model_selection.KFold(n_splits=folds, shuffle=True, random_state=42)

    # iterate over all folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        
        # get training and validation data using folds
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_valid = df.iloc[val_idx].reset_index(drop=True)

        # get the tweets and targets
        x_train = df_train["tweet"].values
        y_train = df_train["target"].values
        x_valid = df_valid["tweet"].values
        y_valid = df_valid["target"].values

        # fitting tokenizer on training data
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(x_train)
        x_train = tokenizer.texts_to_sequences(x_train)
        x_valid = tokenizer.texts_to_sequences(x_valid)

        # pad the sequences with zeros
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=config.MAX_LEN) 
        x_valid = tf.keras.preprocessing.sequence.pad_sequences(x_valid, maxlen=config.MAX_LEN)

        # initialize the dataset class for training and validation
        train_dataset = dataset.TweetDataset(
            tweets=x_train,
            targets=y_train
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            num_workers=2
        )
        valid_dataset = dataset.TweetDataset(
            tweets=x_valid,
            targets=y_valid
        )
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            num_workers=1
        )

        # load embeddings
        print("Loading embeddings...")
        embedding_dict = load_vectors(config.W2V_EMBEDDING_FILE)
        embedding_matrix = create_embedding_matrix(
            tokenizer.word_index,
            embedding_dict
        )

        device = torch.device("cuda")

        # initialize the model
        print("Initializing the model...")
        model = models.LSTM(embedding_matrix)
        model.to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print(f"Training for fold {fold + 1} ...")
        # set best accuracy to zero
        best_accuracy = 0
        # set early stopping counter to zero
        early_stopping_counter = 0
        for epoch in range(config.EPOCHS):
            # train one epoch
            train(train_data_loader, model, optimizer, device)
            # validate
            outputs, targets = evaluate(valid_data_loader, model, device)
            # get the accuracy
            outputs = np.where(np.array(outputs) >= 0.5, 1, 0)
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score = metrics.f1_score(targets, outputs)
            # print accuracy
            print(f"Epoch={epoch}, Accuracy={accuracy}, F1 Score={f1_score}")
            # check if accuracy is greater than the best accuracy
            if accuracy > best_accuracy:
                # update best accuracy
                best_accuracy = accuracy
                # save the model to pickle file
                torch.save(model, config.LSTM_MODEL_PATH)
            else:
                # increment early stopping counter
                early_stopping_counter += 1
            # check if early stopping counter reaches 2
            if early_stopping_counter > 2:
                # break the loop
                break

def predict(tweet_lists, model, device):
    """
    Function for evaluating the model
    :tweet_lists: list of tweets
    :param model: torch model(lstm, cnn, etc.)
    :param device: cuda or cpu
    """

    final_predictions = []

    # initialize the dataloader
    test_dataset = dataset.TweetDataset(
        tweets=tweet_lists,
        targets=[0]*len(tweet_lists)
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # put the model in eval mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():
        for data in test_data_loader:
            tweets = data["tweet"]
            tweets = tweets.to(device, dtype=torch.long)

            # make predictions
            predictions = model(tweets)

            # move predictions and targets to list
            predictions = predictions.cpu().numpy().tolist()
            predictions = np.where(np.array(predictions) >= 0.5, 1, 0)
            # map predictions to -1 and 1
            final_predictions = np.where(final_predictions == 0, -1, 1)
            final_predictions.extend(predictions)
    final_predictions = np.array(final_predictions).flatten()
    # return final predictions and targets
    return final_predictions