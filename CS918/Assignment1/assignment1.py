import json
import re
import nltk
from nltk import trigrams
from collections import defaultdict
import operator
import math

# ============Beginning of Part A============

articles = []
lemmatised = []  # All of lemmatised words
lemmatised_first_16000 = []  # Lemmatised words for the first 16000 articles
lemmatised_after_16000 = []  # Lemmatised words for the rest of the articles
pos_words = {}  # Positive words for sentiment analysis
neg_words = {}  # Negative words for sentiment analysis

with open('signal-news1/signal-news1.jsonl', 'r') as f:
    for line in f:
        tmp_article = {'content': json.loads(line)['content']}
        tmp_article['content'] = tmp_article['content'].lower()
        # Remove Url
        tmp_article['content'] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tmp_article['content'], flags=re.MULTILINE)
        # Remove non-alphanumeric except spaces
        tmp_article['content'] = re.sub(r'[^a-zA-Z\d\s:]', '', tmp_article['content'], flags=re.MULTILINE)
        # Remove single characters
        tmp_article['content'] = re.sub(r'\b(\w)\b', '', tmp_article['content'], flags=re.MULTILINE)
        # Remove single numbers
        tmp_article['content'] = re.sub(r'\b\d+\b', '', tmp_article['content'], flags=re.MULTILINE)
        articles.append(tmp_article)

articles_count = 0
for article in articles:
    words = article['content'].split()
    article['lemmatised'] = {}
    articles_count += 1
    for word in words:
        lemm = nltk.stem.WordNetLemmatizer().lemmatize(word)  # Lemmatise the word
        if articles_count <= 16000:
            lemmatised_first_16000.append(lemm)
        else:
            lemmatised_after_16000.append(lemm)
        if lemm not in article['lemmatised']:
            article['lemmatised'][lemm] = 1
        else:
            article['lemmatised'][lemm] += 1
lemmatised = lemmatised_first_16000 + lemmatised_after_16000

# ============End of Part A============

# ============Beginning of Part B============

print('Number of Tokens (N): ', len(lemmatised))
print('Vocabulary Size (V): ', len(set(lemmatised)))

# Calculate top 25 trigrams
tri = trigrams(lemmatised)
# top = tri.most_common(25)
dist = {}
for g in tri:
    if g in dist:
        dist[g] += 1
    else:
        dist[g] = 1
top25 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:25]  # Sorting trigrams and selecting top 25
print('Top 25 trigrams: ', [g[0] for g in top25])  # Selecting the key (Trigram tuple). The frequency is g[1]

# Load positive and negative words
with open('signal-news1/opinion-lexicon-English/positive-words.txt') as f:
    for line in f:
        pos_words[line.strip()] = 1

with open('signal-news1/opinion-lexicon-English/negative-words.txt') as f:
    for line in f:
        neg_words[line.strip()] = -1


# Count positive words in a word set
def count_pos_words(words_set):
    num = 0
    for w in words_set:
        if w in pos_words:
            num += words_set[w]
    return num


# Count negative words in a word set
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

# ============End of Part B============

# ============Beginning of Part C============

# Generate a 10 word sentence
model = defaultdict(lambda: defaultdict(lambda: 0.01))  # For smoothing
first_16000_trigrams = trigrams(lemmatised_first_16000, pad_right=True, pad_left=True)

for w1, w2, w3 in first_16000_trigrams:
    model[(w1, w2)][w3] += 1  # Count the appearance of the word

for pair in model:
    total_count = float(sum(model[pair].values()))
    for w3 in model[pair]:
        model[pair][w3] /= total_count  # Count the probability of the word appearing

sentence = ["is", "this"]  # First 2 words of the sentence
while len(sentence) < 10:
    words = model.get(tuple(sentence[-2:]))  # Get the trigrams starting with the last pair of words
    word = max(words.items(), key=operator.itemgetter(1))[0]  # Get the next word with maximum probability
    sentence.append(word)
print('Generated 10 word sentence:', ' '.join([w for w in sentence if w]))

# Calculate the perplexity
P = 0
N = 0
for w1, w2, w3 in trigrams(lemmatised_after_16000, pad_left=True, pad_right=True):
    N += 1
    P += math.log2(model[(w1, w2)][w3])
perplexity = pow((1 / abs(P)), 1 / float(N))
print("Perplexity: ", perplexity)

# ============End of Part C============
