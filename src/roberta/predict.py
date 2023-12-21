# predict.py
import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import ROBERTAClassifier

def predict():
    # read the test data
    data = pd.read_csv(config.TEST_FILE, sep="\t")
    # initialize the dataset and dataloader
    test_dataset = dataset.ROBERTATestDataset(
        tweets=data.tweet.values,
        lengths=data.length.values
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # initialze the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # load the model and send it to device
    model = ROBERTAClassifier()
    model.to(device)

    # load the trained model
    model.load_state_dict(torch.load(config.MODEL_FOLDER + str(3)+ config.MODEL_PATH))
    # put the model in evaluation mode
    model.eval()

    # initialize the output list
    final_output = engine.inference_fn(test_data_loader, model, device)

    # return the final_output
    return final_output

if __name__ == "__main__":
    preds = predict()
    submission = pd.DataFrame({'Id':range(1, len(preds) + 1),'Prediction': preds})
    submission.to_csv(config.PREDICTION_PATH, index=False)