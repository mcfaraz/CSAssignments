'''
Title: CS918 Assignment 2
Author: Faraz Taheri (u1534783)
'''

import time



import re
import nltk
from gensim.models import KeyedVectors
from nltk.corpus import stopwords  # TODO: Add stopwords and emojis
from nltk.corpus import sentiwordnet as SWNet
import sklearn.feature_extraction.text as skFeatExt
from sklearn.neural_network import MLPClassifier
import numpy as np

nltk.data.path.append('/modules/cs918/nltk_data/')  # For DCS Workstations
nltk.download('sentiwordnet')  # TODO: Download other lists
model_path = "/Users/Faraz/Developer/CSAssignments/CS918/Assignment2/word2vec_twitter_model/word2vec_twitter_model.bin"
start = time.time()
print("Loading the model, this can take some time...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
print("Loaded the model in: ", time.time()-start)
#print("The vocabulary size is: "+str(len(model.vocab)))

# ============Beginning of pre-processing============
tweets = []
lemmatised_tweets = []
pos_tagged_tweets = []
pos_words = {}  # Positive words for sentiment analysis
neg_words = {}  # Negative words for sentiment analysis
ugram_vectorizer = None

# Load positive and negative words
with open('positive-words.txt') as f:
    for line in f:
        pos_words[line.strip()] = 1

with open('negative-words.txt') as f:
    for line in f:
        neg_words[line.strip()] = -1


def preprocess(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            loaded_line = line.split()
            tmp_tweet = dict()
            tmp_tweet['id'] = loaded_line[0]
            tmp_tweet['sentiment'] = loaded_line[1]
            tmp_tweet['content'] = ' '.join(loaded_line[2:])
            tmp_tweet['content'] = re.sub(
                r'(http)[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                ' urlink ', tmp_tweet['content'], flags=re.MULTILINE)  # Remove Url

            tmp_tweet['content'] = re.sub(r'\s([@][\w_-]+)', ' usrmnt ', tmp_tweet['content'],
                                          flags=re.MULTILINE | re.IGNORECASE)  # Replace mentions
            tmp_tweet['content'] = re.sub(r"(.)\1+", r"\1\1", tmp_tweet['content'])  # Replace elongated words
            # TODO Hashtags, Mentions
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


def build_binary_unigrams():
    global ugram_vectorizer
    # TODO: Change the method
    # TODO: Bigrams
    ugram_vectorizer = skFeatExt.CountVectorizer()
    tweets_list = [t['lemmatised'] for t in tweets[:50]]
    tweets_str = []
    for t in tweets_list:
        tweets_str.append(' '.join(t))
    X = ugram_vectorizer.fit_transform(tweets_str)
    #print(ugram_vectorizer.get_feature_names())
    WVArray = X.toarray()
    return WVArray


def get_binary_unigram(new_tweets):
    # TODO: Change the method
    # TODO: Bigrams
    if not isinstance(new_tweets, (list,)):
        new_tweets = [new_tweets]
    tweets_list = [t['lemmatised'] for t in new_tweets]
    tweets_str = []
    for t in tweets_list:
        tweets_str.append(' '.join(t))
    return ugram_vectorizer.transform(tweets_str).toarray()


def get_word_embedding_score(tweet):

    '''
    vector = []
    for word in tweet['content'].split():
        try:
            vector.append(model.wv[word])
        except KeyError:
            "Word doesn't exist!!!"'''

    sent_vec = []
    numw = 0
    for w in tweet['lemmatised']:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except KeyError:
            # "Word doesn't exist!!!"
            pass

    return np.asarray(sent_vec) / numw


# Count positive words in a lemmatised word set
def count_pos_words(tweet):  # Lemmatised
    num = 0
    for word in tweet:
        if word in pos_words:
            num += 1
    return num


# Count negative words in a lemmatised word set
def count_neg_words(tweet):  # Lemmatised
    num = 0
    for word in tweet:
        if word in neg_words:
            num += 1
    return num


def count_pos_neg_score(tweet):  # Lemmatised
    return {'pos': count_pos_words(tweet), 'neg': count_neg_words(tweet)}


def count_lexicon_score(tweet):  # Lemmatised
    pos_conv_type = {'N': 'n', 'V': 'v', 'J': 'a', 'R': 'r'}  # Key: NLTK pos-tags first letter, Value: Sentiword tags
    scores = {'pos': 0, 'neg': 0, 'obj': 0}
    pos_tagged_tweet = nltk.pos_tag(tweet)

    for pos in pos_tagged_tweet:
        if pos[1][0] in pos_conv_type:
            swnet_pos_type = pos_conv_type[pos[1][0]]
            swnet_lookup_word = '{}.{}.01'.format(pos[0], swnet_pos_type)
            try:
                lex = SWNet.senti_synset(swnet_lookup_word)
                scores['pos'] += lex.pos_score()
                scores['neg'] += lex.neg_score()
                scores['obj'] += lex.obj_score()
            except nltk.corpus.reader.wordnet.WordNetError:
                continue
    return scores


def count_tweet_score(tweet):
    return {'lex_score': count_lexicon_score(tweet['lemmatised']), 'pos_neg_score': count_pos_neg_score(tweet['lemmatised'])}

if __name__ == "__main__":
    preprocess('semeval-tweets/twitter-training-data.txt')
    #print('Vocab size:', len(set(lemmatised_tweets)))
    #print('Tweets: ', len(tweets))
    #for tweet in tweets:
    #    print(count_lexicon_score(tweet['lemmatised']))
    WVArray = build_binary_unigrams()
    start = time.time()
    build_binary_unigrams()
    i = 0
    X_Train = []
    Y_Train = []
    for tweet in tweets[:50]:
        features = []
        lexicon_score = count_lexicon_score(tweet['lemmatised'])
        features.extend(lexicon_score.values())
        tweet['binary_unigram'] = WVArray[i]
        features.extend(tweet['binary_unigram'].tolist())
        features.extend(get_word_embedding_score(tweet).tolist())

        X_Train.append(features)
        Y_Train.append(tweet['sentiment'])
        i += 1

    #print(X_Train)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
    clf.fit(X_Train, Y_Train)
    to_predict = tweets[5001:5005]
    print(to_predict)
    for tweet in to_predict:
        features = []
        lexicon_score = count_lexicon_score(tweet['lemmatised'])
        features.extend(lexicon_score.values())
        tweet['binary_unigram'] = get_binary_unigram(tweet)[0]
        features.extend(tweet['binary_unigram'].tolist())
        features.extend(get_word_embedding_score(tweet).tolist())

        print(clf.predict([features]))

    '''MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                  beta_1=0.9, beta_2=0.999, early_stopping=False,
                  epsilon=1e-08, hidden_layer_sizes=(5, 3),
                  learning_rate='constant', learning_rate_init=0.001,
                  max_iter=200, momentum=0.9, n_iter_no_change=10,
                  nesterovs_momentum=True, power_t=0.5, random_state=1,
                  shuffle=True, solver='lbfgs', tol=0.0001,
                  validation_fraction=0.1, verbose=False, warm_start=False)
                  '''

print(time.time() - start)
