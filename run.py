import sys
sys.path.append("./src/roberta/")

import config
import dataset
import engine
import model
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from statistics import mode

import transformers
from transformers import get_linear_schedule_with_warmup


class ROBERTAClassifier(nn.Module):
    def __init__(self):
        super(ROBERTAClassifier, self).__init__()
        self.roberta = transformers.AutoModelForSequenceClassification.from_pretrained(config.TWITTER_ROBERTA_MODEL)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(4, 1)

    def forward(self, ids, mask,lengths):
        sequence_output = self.roberta(
            ids,
            attention_mask=mask
        )[0]
        # shape of sequence_output: (batch_size, 3)
        output = self.drop(sequence_output)
        # concatenate the lengths to the output, shape: (batch_size, 4)
        output = torch.cat((output, lengths.unsqueeze(1)), dim=1)
        # linear output layer without Sigmoid
        output = self.out(output)
        return output

def predict(model_path):
    # read the test data
    data = pd.read_csv('./data/test.tsv', sep="\t")
    # initialize the dataset and dataloader
    test_dataset = dataset.ROBERTATestDataset(
        tweets=data.tweet.values,
        lengths=data.length.values
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # initialze the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # load the model and send it to device
    model = ROBERTAClassifier()
    model.to(device)

    # load the trained model
    model.load_state_dict(torch.load(model_path))
    # put the model in evaluation mode
    model.eval()

    # initialize the output list
    final_output = engine.inference_fn(test_data_loader, model, device)

    # return the final_output
    return final_output

if __name__== "__main__":
    preds_1 = predict('./manipulated/RoBERTa/1fold_model.bin')
    preds_2 = predict('./manipulated/RoBERTa/2fold_model.bin')
    preds_3 = predict('./manipulated/RoBERTa/3fold_model.bin')
    preds_4 = predict('./manipulated/RoBERTa/4fold_model.bin')
    preds_5 = predict('./manipulated/RoBERTa/5fold_model.bin')
    all_predictions = [preds_1, preds_2, preds_3, preds_4, preds_5]
    final_preds = [mode(votes) for votes in zip(*all_predictions)]

    submission = pd.DataFrame({'Id':range(1, len(final_preds) + 1),'Prediction': final_preds})
    submission.to_csv('submission.csv', index=False)