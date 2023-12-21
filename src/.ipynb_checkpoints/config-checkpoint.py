# config.py
import transformers

# Maximum numbers of tokens in the sentence, in our case we choose 200
MAX_LEN = 200

# Training stage
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

EPOCH = 2

# Model
TWITTER_ROBERTA_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_PATH = "fold_model.bin"

# Training file
TRAINING_FILE = "../data/twitter-datasets/train_kfold.tsv"

# Tokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(TWITTER_ROBERTA_MODEL)