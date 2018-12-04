'''
Title: CS918 Assignment 2
Author: Faraz Taheri (u1534783)
'''

import json
import re
from collections import defaultdict
import operator
import math
import nltk
from nltk import trigrams
import sklearn.feature_extraction.text as skFeatExt
nltk.data.path.append('/modules/cs918/nltk_data/')  # For DCS Workstations

import sys
sys.path.append('/Users/Faraz/Developer/CSAssignments/CS918/Assignment2/word2vec_twitter_model')
import word2vecReader

# ============Beginning of pre-processing============
tweets = []
lemmatised_tweets = []


def preprocess(file_name):
    # The signal-news1 folder must be located in the same directory as this file
    with open(file_name, 'r') as f:
        for line in f:
            loaded_line = line.split()
            tmp_tweet = dict()
            tmp_tweet['id'] = loaded_line[0]
            tmp_tweet['sentiments'] = loaded_line[1]
            tmp_tweet['content'] = ' '.join(loaded_line[2:])
            # Remove Url
            tmp_tweet['content'] = re.sub(
                r'(http)[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                ' urlink ', tmp_tweet['content'], flags=re.MULTILINE)

            tmp_tweet['content'] = re.sub(r'\s([@][\w_-]+)', ' usrmnt ', tmp_tweet['content'],
                                          flags=re.MULTILINE | re.IGNORECASE)  # Replace mentions
            tmp_tweet['content'] = re.sub(r"(.)\1+", r"\1\1", tmp_tweet['content'])  # Replace elongated words
            # TODO Hashtags
            tmp_tweet['content'] = re.sub(r'[^a-zA-Z\d\s]', ' ', tmp_tweet['content'],
                                          flags=re.MULTILINE)  # Remove non-alphanumeric except spaces
            tmp_tweet['content'] = re.sub(r'\b(\w)\b', '', tmp_tweet['content'],
                                          flags=re.MULTILINE)  # Remove single characters
            tmp_tweet['content'] = re.sub(r'\b\d+\b', '', tmp_tweet['content'],
                                          flags=re.MULTILINE)  # Remove single numbers
            tmp_tweet['content'] = re.sub(r'\b[A-Z]+\b', lambda m: 'uppercase' + m.group(0),
                                          tmp_tweet['content'],
                                          flags=re.MULTILINE)  # Replace upper-case words

            tmp_tweet['content'] = tmp_tweet['content'].lower()
            tmp_tweet['lemmatised'] = []
            tweets.append(tmp_tweet)

    for t in tweets:
        words = t['content'].split()
        for word in words:
            lemm = nltk.stem.WordNetLemmatizer().lemmatize(word)  # Lemmatise the word
            lemmatised_tweets.append(lemm)
            t['lemmatised'].append(lemm)

# ============End of pre-processing============


def feature_extraction():
    # Vectorizer
    for t in tweets:
        vectorizer = skFeatExt.CountVectorizer()
        X = vectorizer.fit_transform(t['lemmatised'])
        print(vectorizer.get_feature_names())
        #print(X)
        print(X.toarray())

    #print('Hash')
    #vectorizer = skFeatExt.HashingVectorizer(n_features=2 ** 4)
    #X = vectorizer.fit_transform(lemmatised_tweets)
    #print(X)
    #print(X.shape)


if __name__ == "__main__":
    preprocess('semeval-tweets/twitter-test1.txt')
    feature_extraction()