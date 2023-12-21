# train.py
import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import ROBERTAClassifier
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def train(fold):
    # read the n-th fold training data
    data = pd.read_csv(config.TRAINING_FILE, sep="\t")
    # keep 30% of the data
    data = data.sample(frac=0.03).reset_index(drop=True)
    data_train = data[data.fold != fold].reset_index(drop=True)
    data_valid = data[data.fold == fold].reset_index(drop=True)

    # initialize the dataset and dataloader
    train_dataset = dataset.ROBERTADataset(
        tweets=data_train.tweet.values,
        labels=data_train.label.values,
        lengths=data_train.length.values
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )
    valid_dataset = dataset.ROBERTADataset(
        tweets=data_valid.tweet.values,
        labels=data_valid.label.values,
        lengths=data_valid.length.values
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # initialze the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # load the model and send it to device
    model = ROBERTAClassifier()
    model.to(device)

    # load the trained model
    # model.load_state_dict(torch.load("../manipulated/RoBERTa/3fold_model.bin"))
    # put the model in evaluation mode
    model.eval()

    # set parameters for the optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # calculate the training steps
    num_train_steps = int(len(data_train) / config.TRAIN_BATCH_SIZE * config.EPOCH)

    # initialize the optimizer
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    # initialize the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # start the training
    best_accuracy = 0
    for epoch in range(config.EPOCH):
        print(f"Training for Fold: {fold}, Epoch: {epoch}")
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score = metrics.f1_score(targets, outputs)
        print(f"Fold: {fold}, Epoch: {epoch}, Accuracy Score = {accuracy}, F1 Score = {f1_score}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_FOLDER + str(fold)+ config.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == "__main__":
    # train(1)
    train(3)
    # train(3)
    # train(4)
    # train(5)