# engine.py
import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
from tqdm import tqdm

def loss_fn(outputs, targets):
    """
    :param outputs: outputs from the model, real numbers
    :param targets: targets from the dataset, binary numbers
    """
    loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
    return loss

def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    This is the training function for one epoch
    :param data_loader: torch dataloader object
    :param model: torch model object
    :param optimizer: adam, sgd, etc.
    :param device: cuda/cpu
    :param scheduler: learning rate scheduler
    """

    # put the model in training mode
    model.train()

    # loop over all batches
    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        # extract ids, mask, token_type_ids, targets and lengths from the data dictionary
        ids = data["ids"]
        mask = data["mask"]
        targets = data["targets"]
        lengths = data["lengths"]

        # move all the variables to GPU if available
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        lengths = lengths.to(device, dtype=torch.float)

        # zero-grad the optimizer
        optimizer.zero_grad()

        # pass the inputs to the model
        outputs = model(
            ids=ids,
            mask=mask,
            lengths=lengths
        )

        loss = loss_fn(outputs, targets)

        # backward pass
        loss.backward()

        # step the optimizer and scheduler
        optimizer.step()
        scheduler.step()

def eval_fn(data_loader, model, device):
    """
    This is the evaluation function for one epoch
    :param data_loader: torch dataloader object
    :param model: torch model object
    :param device: cuda/cpu
    :return: fin_outputs, fin_targets
    """

    # put the model in evaluation mode
    model.eval()

    # init lists to store targets and outputs
    fin_targets = []
    fin_outputs = []

    # use no_grad to save memory during evaluation
    with torch.no_grad():
        # loop over all batches
        for batch_idx, data in enumerate(data_loader):
            # extract ids, mask, token_type_ids, targets and lengths from the data dictionary
            ids = data["ids"]
            mask = data["mask"]
            targets = data["targets"]
            lengths = data["lengths"]

            # move all the variables to GPU if available
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            lengths = lengths.to(device, dtype=torch.float)

            # pass the inputs to the model
            outputs = model(
                ids=ids,
                mask=mask,
                lengths=lengths
            )

            # convert outputs to sigmoid
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()

            # extend the original list
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs)

    return fin_outputs, fin_targets

def inference_fn(data_loader, model, device):
    """
    This is the inference function
    :param data_loader: torch dataloader object
    :param model: torch model object
    :param device: cuda/cpu
    :return: fin_outputs, which is a list of binary numbers
    """

    # put the model in evaluation mode
    model.eval()

    # init lists to store outputs
    fin_outputs = []

    # use no_grad to save memory during evaluation
    with torch.no_grad():
        # loop over all batches
        for batch_idx, data in enumerate(data_loader):
            # extract ids, mask, token_type_ids, targets and lengths from the data dictionary
            ids = data["ids"]
            mask = data["mask"]
            lengths = data["lengths"]

            # move all the variables to GPU if available
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            lengths = lengths.to(device, dtype=torch.float)

            # pass the inputs to the model
            outputs = model(
                ids=ids,
                mask=mask,
                lengths=lengths
            )

            # convert outputs to sigmoid
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()

            # extend the original list
            fin_outputs.extend(outputs)

        # flatten the list
        fin_outputs = np.array(fin_outputs).flatten()
        # convert the final output to binary numbers
        fin_outputs = np.array(fin_outputs) >= 0.5
        # map the binary numbers to -1 and 1
        fin_outputs = np.where(fin_outputs == 0, -1, 1)

    return fin_outputs