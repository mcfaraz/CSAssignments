import json
import re
import nltk
from nltk import bigrams, trigrams, ngrams
import time
from collections import defaultdict
import random

# ============Beginning of Part A============

start_time = time.time()
articles = []
lemmatised = []
lemmatised_16000 = []
pos_words = {}
neg_words = {}

i = 0  # TODO: Remove this
with open('signal-news1/signal-news1.jsonl', 'r') as f:
    for line in f:
        tmp_article = {}
        tmp_article['content'] = json.loads(line)['content']
        tmp_article['content'] = tmp_article['content'].lower()
        # All Combined: (http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|([^a-zA-Z\d\s:])|(\b(\w)\b)|(\b\d+\b)
        # Remove Url
        tmp_article['content'] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tmp_article['content'], flags=re.MULTILINE)
        # Remove non-alphanumeric except spaces
        tmp_article['content'] = re.sub(r'[^a-zA-Z\d\s:]', '', tmp_article['content'], flags=re.MULTILINE)
        # Remove single characters
        tmp_article['content'] = re.sub(r'\b(\w)\b', '', tmp_article['content'], flags=re.MULTILINE)
        # Remove single numbers
        tmp_article['content'] = re.sub(r'\b\d+\b', '', tmp_article['content'], flags=re.MULTILINE)
        articles.append(tmp_article)

        #i = i + 1
        #if (i > 10):
        #    break

# print('Loading corpus took: ', time.time() - start_time)
# start_time = time.time()

i = 0
for article in articles:
    words = article['content'].split()
    article['lemmatised'] = {}
    i += 1
    for word in words:
        lemm = nltk.stem.WordNetLemmatizer().lemmatize(word)
        lemmatised.append(lemm)
        if i <= 16000:
            lemmatised_16000.append(lemm)
        if lemm not in article['lemmatised']:
            article['lemmatised'][lemm] = 1
        else:
            article['lemmatised'][lemm] += 1

# print('Lemmatisation took: ', time.time() - start_time)
# start_time = time.time()

# ============End of Part A============


# ============Beginning of Part B============

print('Number of Tokens (N): ', len(lemmatised))
print('Vocabulary Size (V): ', len(set(lemmatised)))

# Calculating top 25 trigrams
tri = trigrams(lemmatised)
# top = tri.most_common(25)
dist = {}
for g in tri:
    if g in dist:
        dist[g] += 1
    else:
        dist[g] = 1
top25 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:25]  # Sorting trigrams and selecting top 25
print('Top 25 trigrams: ', [g[0] for g in top25])
# print('Top 25 took: ', time.time() - start_time)
# start_time = time.time()

# Load positive and negative words
with open('signal-news1/opinion-lexicon-English/positive-words.txt') as f:
    for line in f:
        pos_words[line.strip()] = 1

with open('signal-news1/opinion-lexicon-English/negative-words.txt') as f:
    for line in f:
        neg_words[line.strip()] = -1


def count_pos_words(words_set):
    num = 0
    for w in words_set:
        if w in pos_words:
            num += words_set[w]
    return num


def count_neg_words(words_set):
    num = 0
    for w in words_set:
        if w in neg_words:
            num += words_set[w]
    return num


num_pos_articles = 0
num_neg_articles = 0
total_pos_words = 0
total_neg_words = 0

for article in articles:
    num_pos_words = count_pos_words(article['lemmatised'])
    num_neg_words = count_neg_words(article['lemmatised'])
    total_pos_words += num_pos_words
    total_neg_words += num_neg_words

    if num_pos_words > num_neg_words:
        num_pos_articles += 1
    elif num_pos_words < num_neg_words:
        num_neg_articles += 1

print('Number of positive words: ', total_pos_words)
print('Number of negative words: ', total_neg_words)
print('Number of positive articles: ', num_pos_articles)
print('Number of negative articles: ', num_neg_articles)
# print('Counting took: ', time.time() - start_time)
# start_time = time.time()

# ============End of Part B============

# ============Beginning of Part C============

def trigram_lang_model():
    model = []

    first_16000_trigrams = trigrams(lemmatised_16000, pad_right=True, pad_left=True)
    for f in first_16000_trigrams:
        model.append(f)
    return model


first_16000_trigrams = trigrams(lemmatised_16000, pad_right=True, pad_left=True)
model = defaultdict(lambda: defaultdict(lambda: 0))
for w1, w2, w3 in first_16000_trigrams:
    model[(w1, w2)][w3] += 1
# Let's transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
text = ["is", "this"]
#text = [None, None]

sentence_finished = False

#while not sentence_finished and len(text) < 12:
while len(text) < 3:
    r = random.random()
    accumulator = .0

    for word in model[tuple(text[-2:])].keys():
        accumulator += model[tuple(text[-2:])][word]

        if accumulator >= r:
            text.append(word)
            break

    if text[-2:] == [None, None]:
        sentence_finished = True
    #print(' '.join([t for t in text if t]))

print(' '.join([t for t in text if t]))


# ============End of Part C============
