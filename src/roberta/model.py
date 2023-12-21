# model.py
import config
import transformers
import torch
import torch.nn as nn

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