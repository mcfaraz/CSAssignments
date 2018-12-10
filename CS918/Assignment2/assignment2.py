'''
Title: CS918 Assignment 2
Author: Faraz Taheri (u1534783)
'''

import time
import re
import nltk
from gensim.models import KeyedVectors
from nltk.corpus import stopwords  # TODO: Add emojis
from nltk.corpus import sentiwordnet as SWNet
import sklearn.feature_extraction.text as skFeatExt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

nltk.data.path.append('/modules/cs918/nltk_data/')  # For DCS Workstations
nltk.download('sentiwordnet')  # TODO: Download other lists
en_stop_words = set(stopwords.words('english'))
model_path = "/Users/Faraz/Developer/CSAssignments/CS918/Assignment2/word2vec_twitter_model/word2vec_twitter_model.bin"

start = time.time()
print("Loading the twitter Word2Vector model")
model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
print("Loaded the model in: {} sec".format(time.time()-start))

# ============Beginning of pre-processing============
pos_words = {}  # Positive words for sentiment analysis
neg_words = {}  # Negative words for sentiment analysis

def load_sentiment_words():
    with open('positive-words.txt') as f:
        for line in f:
            pos_words[line.strip()] = 1

    with open('negative-words.txt') as f:
        for line in f:
            neg_words[line.strip()] = -1


class SentimentAnalyzer:
    # Load positive and negative words
    def __init__(self, file_name):
        self.pos_tagged_tweets = []
        self.ugram_vectorizer = None
        self.tweets = self.preprocess(file_name)
        self.trained_model = None

    @staticmethod
    def preprocess(file_name):
        with open(file_name, 'r') as f:
            raw_tweets = []
            for line in f:
                loaded_line = line.split()
                tmp_tweet = dict()
                tmp_tweet['id'] = loaded_line[0]
                tmp_tweet['sentiment'] = loaded_line[1]
                tmp_tweet['content'] = ' '.join(loaded_line[2:])
                tmp_tweet['content'] = re.sub(
                    r'(http)[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                    ' urlink ', tmp_tweet['content'], flags=re.MULTILINE)  # Remove Url
                tmp_tweet['num_mentions'] = len(re.findall(r'\s[@][\w_]+', tmp_tweet['content'])) # Number of mentions
                tmp_tweet['num_exclm_marks'] = len(re.findall(r'!', tmp_tweet['content']))  # Number of exclamation marks
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
                #tmp_tweet['uppercase_words'] = len(re.findall(r'\b[A-Z]+\b', tmp_tweet['content']))  # Count uppercases
                tmp_tweet['content'] = re.sub(r'\b[A-Z]+\b', lambda m: 'uppercase' + m.group(0),
                                              tmp_tweet['content'],
                                              flags=re.MULTILINE)  # Replace upper-case words

                tmp_tweet['content'] = tmp_tweet['content'].lower()
                tmp_tweet['lemmatised'] = []
                raw_tweets.append(tmp_tweet)

        for t in raw_tweets:
            words = t['content'].split()
            for word in words:
                if word not in en_stop_words:
                    lemm = nltk.stem.WordNetLemmatizer().lemmatize(word)  # Lemmatise the word
                    t['lemmatised'].append(lemm)

        return raw_tweets

    def build_binary_unigrams(self):
        global ugram_vectorizer
        # TODO: Change the method
        # TODO: Bigrams
        #ugram_vectorizer = skFeatExt.CountVectorizer()
        ugram_vectorizer = skFeatExt.TfidfVectorizer(min_df=5)
        tweets_list = [t['lemmatised'] for t in self.tweets]
        tweets_str = []
        for t in tweets_list:
            tweets_str.append(' '.join(t))
        X = ugram_vectorizer.fit_transform(tweets_str)
        #print(ugram_vectorizer.get_feature_names())
        WVArray = X.toarray()
        return WVArray

    @staticmethod
    def get_binary_unigram(new_tweets):
        # TODO: Change the method
        # TODO: Bigrams
        if not isinstance(new_tweets, (list,)):
            new_tweets = [new_tweets]
        tweets_list = [t['lemmatised'] for t in new_tweets]
        tweets_str = []
        for tweet in tweets_list:
            words = [word for word in tweet if word not in en_stop_words]
            tweets_str.append(' '.join(words))
        return ugram_vectorizer.transform(tweets_str).toarray()

    @staticmethod
    def get_word_embedding_score(tweet):
        sent_vec = []
        num_words = 0
        for w in tweet['lemmatised']:
            try:
                if num_words == 0:
                    sent_vec = model[w]
                else:
                    sent_vec = np.add(sent_vec, model[w])
                    num_words += 1
            except KeyError:
                # "Word doesn't exist!!!"
                pass
        #return np.mean(sent_vec)
        ##ret = np.asarray(sent_vec) / numw
        return [np.mean(sent_vec)] if len(sent_vec) > 0 else [0]

    # Count positive words in a lemmatised word set
    @staticmethod
    def count_pos_words(tweet):
        num = 0
        for word in tweet:
            if word in pos_words:
                num += 1
        return num

    # Count negative words in a lemmatised word set
    @staticmethod
    def count_neg_words(tweet):
        num = 0
        for word in tweet:
            if word in neg_words:
                num += 1
        return num

    def count_pos_neg_score(self, tweet):  # Lemmatised
        return [self.count_pos_words(tweet), self.count_neg_words(tweet)]

    @staticmethod
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

    def train_model(self, classifier, save_model = False):
        start = time.time()
        if classifier == 'MLP_all':
            WVArray = self.build_binary_unigrams()
            i = 0

        try:
            self.trained_model = pickle.load(open(classifier+'.pickle', "rb"))
            return
        except (OSError, IOError) as e:
            pass

        X_Train = []
        Y_Train = []
        for tweet in self.tweets:
            features = []
            features.extend(self.get_word_embedding_score(tweet))
            lexicon_score = self.count_lexicon_score(tweet['lemmatised'])
            features.extend(lexicon_score.values())
            features.extend(self.count_pos_neg_score(tweet['lemmatised']))
            features.append(tweet['num_mentions'])
            features.append(tweet['num_exclm_marks'])
            # features.append(tweet['uppercase_words'])
            if classifier == 'MLP_all':
                tweet['binary_unigram'] = WVArray[i]
                features.extend(tweet['binary_unigram'].tolist())
                i += 1
            X_Train.append(features)
            Y_Train.append(tweet['sentiment'])
        print('Got features in:', time.time() - start)

        print('Now training ')
        start = time.time()
        # self.trained_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
        self.trained_model = MLPClassifier(solver='lbfgs')
        # clf = RandomForestClassifier(n_estimators=100)
        self.trained_model.fit(X_Train, Y_Train)
        print('Trained in:', time.time() - start)
        if save_model:
            with open(classifier+'.pickle', 'wb') as fp:
                pickle.dump(self.trained_model, fp)
            print('Pickled')
        '''MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                          beta_1=0.9, beta_2=0.999, early_stopping=False,
                          epsilon=1e-08, hidden_layer_sizes=(5, 3),
                          learning_rate='constant', learning_rate_init=0.001,
                          max_iter=200, momentum=0.9, n_iter_no_change=10,
                          nesterovs_momentum=True, power_t=0.5, random_state=1,
                          shuffle=True, solver='lbfgs', tol=0.0001,
                          validation_fraction=0.1, verbose=False, warm_start=False)
                          '''

    def predict_sentiment(self, to_predict, classifier):
        result = {}
        for tweet in to_predict:
            features = []
            features.extend(self.get_word_embedding_score(tweet))
            lexicon_score = self.count_lexicon_score(tweet['lemmatised'])
            features.extend(lexicon_score.values())
            features.extend(self.count_pos_neg_score(tweet['lemmatised']))
            features.append(tweet['num_mentions'])
            features.append(tweet['num_exclm_marks'])
            # features.append(tweet['uppercase_words'])
            if classifier == 'MLP_all':
                tweet['binary_unigram'] = self.get_binary_unigram(tweet)[0]
                features.extend(tweet['binary_unigram'].tolist())
            result[tweet['id']] = self.trained_model.predict([features]).tolist()[0]
            #print(result)
        return result
