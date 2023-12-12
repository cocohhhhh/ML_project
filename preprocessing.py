import re
import emoji
import pandas as pd
import spacy

def remove_tags(text):
    text = text.lower()
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove '<user>'
    text = re.sub(r'<user>', '', text)
    # Remove '<url>'
    text = re.sub(r'<url>', '', text)
    # remove number
    text = re.sub(r'\d+', '', text)
    return text




def convert_emojis(text):
    return emoji.demojize(text)


def remove_urls_and_hashtags(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"#[A-Za-z0-9_]+", '', text)
    return text


# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# Load the urban dictionary dataset
file_path = 'data/urbandict-word-defs.csv'

try:
    slang_df = pd.read_csv(file_path, on_bad_lines='skip')
except AttributeError:
    slang_df = pd.read_csv(file_path, error_bad_lines=False)

# Function to extract meaningful words from definitions
def extract_meaningful_words(definition):
    # Ensure the definition is a string
    if not isinstance(definition, str):
        return ''
    doc = nlp(definition)
    meaningful_words = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ'] and token.text.lower() not in ['word', 'words', 'people', 'something']]
    return ' '.join(meaningful_words)

# Function to preprocess tweets
def replace_slang(tweet, slang_df):
    words = tweet.split()
    processed_words = []

    for word in words:
        # Find the slang definitions in the DataFrame
        word_definitions = slang_df[slang_df['word'] == word.lower()]['definition']

        # Process the definitions
        for definition in word_definitions:
            processed_definition = extract_meaningful_words(definition)
            if len(processed_definition.split()) <= 5:
                processed_words.append(processed_definition)
                break
        else:
            # Append the original word if no suitable definition is found
            processed_words.append(word)

    # Join the words back into a processed tweet
    processed_tweet = ' '.join(processed_words)
    return processed_tweet

# Example tweet
tweet = "Yo, that party last night was lit! Everyone was turnt up, the vibe was on point. The DJ was dropping bangers after bangers, totally slayed. No cap, it was the best night ever. YOLO!"


# Preprocess the example tweet
preprocessed_tweet = replace_slang(tweet, slang_df)

# Output the preprocessed tweet
print("preprocess_tweet", preprocessed_tweet)
