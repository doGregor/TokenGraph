import sys

from nltk.corpus import twitter_samples
import random
import pandas as pd
from datasets import Dataset


def load_twitter_corpus():
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    return positive_tweets, negative_tweets


def samples_to_df(pos_samples, neg_samples):
    pos_samples = [s for s in pos_samples if len(s.strip()) > 0]
    neg_samples = [s for s in neg_samples if len(s.strip()) > 0]

    pos_samples = pos_samples[:150]
    neg_samples = neg_samples[:150]

    pos_labels = [1] * len(pos_samples)
    neg_labels = [0] * len(neg_samples)

    text = pos_samples + neg_samples
    labels = pos_labels + neg_labels
    zipped_lists = list(zip(text, labels))
    random.shuffle(zipped_lists)
    text, labels = zip(*zipped_lists)

    df = pd.DataFrame(list(zip(text, labels)),
                      columns=['text', 'label'])
    return df


def dataset_from_df(df):
    dataset = Dataset.from_pandas(df=df)
    dataset = dataset.train_test_split(test_size=0.2)
    dataset_train, dataset_test = dataset['train'], dataset['test']
    return dataset_train, dataset_test


if __name__ == '__main__':
    positive_tweets, negative_tweets = load_twitter_corpus()
    df = samples_to_df(positive_tweets, negative_tweets)
    dataset_train, dataset_test = dataset_from_df(df)
    print(dataset_train)
    print(dataset_test)

