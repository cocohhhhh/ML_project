import numpy as np
import pickle
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.test.utils import common_texts



pos_path = 'data/twitter-datasets/train_pos.txt'
neg_path = 'data/twitter-datasets/train_neg.txt'

with open(pos_path, 'r') as f:
    pos_tweets = f.readlines()
with open(neg_path, 'r') as f:
    neg_tweets = f.readlines()

# Load the word embeddings
embedding_matrix = np.load('manipulated/glove_embeddings.npy')
# Load the vocabulary
with open('manipulated/vocab.pkl', 'rb') as f:
    vocabulary = pickle.load(f)
# Create a dictionary to map words to their embeddings
embeddings_dict = {word: embedding_matrix[index] for index, word in enumerate(vocabulary)}




###*** Method 1: GloVe ***###

def preprocess(tweet):
    # Assuming tweets are already tokenized and separated by spaces
    return tweet.lower().split()

def tweet_to_vector(tweet):
    words = preprocess(tweet)
    word_vectors = [embeddings_dict[word] for word in words if word in embeddings_dict]
    
    # Handle the case where tweet has no valid words found in embeddings
    if not word_vectors:
        return np.zeros(next(iter(embeddings_dict.values())).shape)
    
    # Compute the average vector
    tweet_vector = np.mean(word_vectors, axis=0)
    return tweet_vector

def glove(tweets):
    tweets = [tweet.rstrip('\n') for tweet in tweets]
    tweet_vectors = np.array([tweet_to_vector(tweet) for tweet in tweets])
    return tweet_vectors




###*** Method 2: CountVectorizer ***###

tweet_tokenizer = TweetTokenizer()
def tokenize_tweets(tweet):
    return tweet_tokenizer.tokenize(tweet)



count_vectorizer = CountVectorizer(tokenizer=tokenize_tweets)

def count_vectorize(tweets):
    tweet_vectors = count_vectorizer.fit_transform(tweets)
    tweet_vectors = tweet_vectors.toarray() # if lacking memory, delete
    return tweet_vectors



###*** Method 3: TF-IDF vectorizer ***###


tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_tweets)
def tfidf_vectorize(tweets):
    tweet_vectors = tfidf_vectorizer.fit_transform(tweets)
    tweet_vectors = tweet_vectors.toarray() # if lacking memory, delete
    return tweet_vectors



###*** Method 4: Word2Vec ***###
model_w2v = Word2Vec(common_texts, vector_size=100, window=5, min_count=1, workers=4)

def word2vec(tweets, model_w2v, vector_size):
    model_w2v.train(tweets, total_examples=len(tweets), epochs=10)
    default_vector = np.zeros(vector_size)
    tweet_vectors = np.array([
        np.mean(
            [model_w2v.wv[word] for word in tweet.split() if word in model_w2v.wv] or [default_vector],
            axis=0
        )
        for tweet in tweets
    ])
    return tweet_vectors

