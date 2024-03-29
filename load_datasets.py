import nltk
import ssl
from nltk.corpus import twitter_samples


def download_twitter_corpus():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('twitter_samples')


def load_twitter_corpus():
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    return positive_tweets, negative_tweets


if __name__ == '__main__':
    positive_tweets, negative_tweets = load_twitter_corpus()
