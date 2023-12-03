
import nltk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import contractions
import demoji

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing
from langdetect import detect, LangDetectException
from nltk.tokenize import word_tokenize

def remove_emojis(text):
    return demoji.replace(text, '')

def strip_all_entities(text):
    text = re.sub(r'\r|\n', ' ', text.lower())
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def clean_hashtags(tweet):
    new_tweet = re.sub(r'(\s+#[\w-]+)+\s*$', '', tweet).strip()
    new_tweet = re.sub(r'#([\w-]+)', r'\1', new_tweet).strip()
    return new_tweet

def filter_chars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

def remove_mult_spaces(text):
    return re.sub(r"\s\s+", " ", text)

def filter_non_english(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""

def expand_contractions(text):
    return contractions.fix(text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def lemmatize(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

def replace_elongated_words(text):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', text)

def remove_repeated_punctuation(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

def remove_extra_whitespace(text):
    return ' '.join(text.split())

def remove_url_shorteners(text):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', text)

def remove_spaces_tweets(tweet):
    return tweet.strip()

def remove_short_tweets(tweet, min_words=3):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""

def clean_tweet(tweet):
    tweet = remove_emojis(tweet)
    tweet = expand_contractions(tweet)
    tweet = filter_non_english(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = remove_numbers(tweet)
    tweet = lemmatize(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = remove_spaces_tweets(tweet)
    tweet = remove_short_tweets(tweet)
    tweet = ' '.join(tweet.split())
    return tweet

def preprocess():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    df = pd.read_csv("/content/drive/MyDrive/MS /ML/cyberbullying_tweets.csv")
    df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})
    df = df[~df.duplicated()]
    df['text_clean'] = [clean_tweet(tweet) for tweet in df['text']]
    df.drop_duplicates("text_clean", inplace=True)
    df = df[df["sentiment"]!="other_cyberbullying"]
    sentiments = ["religion","age","ethnicity","gender","not bullying"]
    df['text_len'] = [len(text.split()) for text in df.text_clean]
    df = df[df['text_len'] < df['text_len'].quantile(0.995)]
    df['sentiment'] = df['sentiment'].replace({'religion':0,'age':1,'ethnicity':2,'gender':3,'not_cyberbullying':4})


if __name__ == "__main__":
    preprocess()