# config.py
import transformers

# Maximum numbers of tokens in the sentence, in our case we choose 200
MAX_LEN = 200

# Training stage
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

EPOCH = 5

# Model
TWITTER_ROBERTA_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
# Trained model path
MODEL_FOLDER = "../manipulated/RoBERTa/"
MODEL_PATH = "fold_model_final.bin"

# Training file
TRAINING_FILE = "../data/train.tsv"

# Test file
TEST_FILE = "../data/test.tsv"
PREDICTION_PATH = "../data/submission_test.csv"

# Tokenizer
TOKENIZER = transformers.RobertaTokenizer.from_pretrained(TWITTER_ROBERTA_MODEL)