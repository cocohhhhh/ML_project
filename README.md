
# Comprehensive Sentiment Analysis Using Different Methods

## Introduction
This project performs comprehensive sentiment analysis, focusing on Twitter data. The goal is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.


## Project Focus
The project compares various sentiment analysis techniques to identify the most effective ones, particularly for Twitter data. It involves:

- Analyzing performance in different scenarios.
- Evaluating accuracy and efficiency.
- Exploring text representation methods like Bag-of-Words, TF-IDF, Word2Vec, and GloVe.

## Data Collection and Preprocessing
Data includes tweets, processed via steps like:

- Cleaning and normalizing text.
- Handling missing values and outliers.
- Utilizing advanced text representation methods.

## Methods Evaluated
The project evaluates:

- Traditional models: Decision Tree, Random Forest, Logistic Regression.
- Advanced models: LSTM, BERT, RoBERTa.

## Training and Evaluation
- Data split into training, validation, and testing sets.
- Emphasis on transformer-based models for their effectiveness.
- Evaluation metrics include accuracy, precision, recall, F1-score, and model robustness.

## Structure of this repository
```Bash
--Data              To store the original dataset
--Manipulated       To store the trained models
--Src               Providing the code we used in this project
run.py              To get the submission file using the best model we find
combined_analysis   Showing the overall workflow of this project.
```


## Dependencies
Dependencies include Python 3.8, Pandas, Numpy, Sklearn, NLTK, TensorFlow, PyTorch.

## Models Used
Models used are Decision Tree, Random Forest, Logistic Regression, LSTM, BERT, RoBERTa.

## Environment & Execution Instructions
- Run in a Python environment, preferably a virtual environment.
- Main execution file is `run.py`.
    - The `run.py` is simply a prediction function of the best model we find (Roberta with 5 fold majority voting). On GPU, the prediction should be given within minutes. To train the model, please go to src folder and check different methods. But training is very time-consuming.
- Can be run in Jupyter Notebook or Google Colab.
