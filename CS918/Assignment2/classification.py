#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import assignment2 as senti

# TODO: load training data
training_data_addr = 'semeval-tweets/twitter-training-data.txt'

for classifier in ['MLP_all', 'myclassifier2', 'myclassifier3']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'MLP_all':
        print('Training ' + classifier)
        clf = senti.SentimentAnalyzer('semeval-tweets/twitter-training-data.txt')
        clf.train_model(classifier)
        #clf.preprocess('semeval-tweets/twitter-training-data.txt')


        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
    elif classifier == 'myclassifier2':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
    elif classifier == 'myclassifier3':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3

    for testset in testsets.testsets:
        # TODO: classify tweets in test set
        print(testset)
        test_tweets = clf.preprocess(testset)
        #predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        predictions = clf.predict_sentiment(test_tweets, classifier)
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)
    break
