import json
import re
import nltk
from nltk import ngrams
from nltk import bigrams, trigrams

# ============Beginning of Part A============

articles = []
lemmatised = []
pos_words = []
neg_words = []


def load_articles():
    i = 0
    with open('signal-news1/signal-news1.jsonl', 'r') as f:
        for line in f:
            tmp_article = json.loads(line)
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
            i = i + 1
            if (i>10):
                break


def lemmatize():
    for article in articles:
        words = article['content'].split()
        article['lemmatised'] = []
        for word in words:
            article['lemmatised'].append(nltk.stem.WordNetLemmatizer().lemmatize(word))
        lemmatised.extend(article['lemmatised'])

# ============End of Part A============


# ============Beginning of Part B============

def num_tokens():
    return len(set(lemmatised))


def vocab_size():
    return len(lemmatised)


def top25():
    tri = trigrams(lemmatised)
    #tri = ngrams(lemmatised, 3)
    #dist = nltk.FreqDist(tri)
    #top = dist.most_common(25)
    dist = {}
    for g in tri:
        if g in dist:
            dist[g] += 1
        else:
            dist[g] = 1
    top25 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:25]
    return top25


def load_words():
    with open('signal-news1/opinion-lexicon-English/positive-words.txt') as f:
        for line in f:
            pos_words.append(line.strip())

    with open('signal-news1/opinion-lexicon-English/negative-words.txt') as f:
        for line in f:
            neg_words.append(line.strip())


def count_pos_words(words):
    num = 0
    for w in words:
        if w in pos_words:
            num = num + 1
    return num


def count_neg_words(words):
    num = 0
    for w in words:
        if w in neg_words:
            num = num + 1
    return num


def analyze_articles():
    num_pos_articles = 0
    num_neg_articles = 0

    for article in articles:
        num_pos_words = count_pos_words(article['lemmatised'])
        num_neg_words = count_neg_words(article['lemmatised'])

        if num_pos_words > num_neg_words:
            num_pos_articles += 1
        elif num_pos_words < num_neg_words:
            num_neg_articles += 1

    return num_pos_articles, num_neg_articles

# ============End of Part B============


# ============Beginning of Part C============
def trigram_lang_model():
    from nltk import bigrams, trigrams


# ============End of Part C============


def main():
    load_articles()
    lemmatize()
    #print(num_tokens())
    #print(vocab_size())
    #print(top25())
    load_words()
    #print('positive')
    #print (num_pos_words())
    #print('negative')
    #print(num_neg_words())
    print(analyze_articles())


main()
