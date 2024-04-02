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


def load_mr_corpus():
    positive_samples = []
    negative_samples = []
    with open('raw_data/MR/rt-polarity.pos', 'rb') as input_data:
        for line in input_data.readlines():
            if len(line.strip()) > 0:
                try:
                    positive_samples.append(line.decode("utf-8").strip().lower())
                except:
                    pass
    with open('raw_data/MR/rt-polarity.neg', 'rb') as input_data:
        for line in input_data.readlines():
            if len(line.strip()) > 0:
                try:
                    negative_samples.append(line.decode("utf-8").strip().lower())
                except:
                    pass
    return positive_samples, negative_samples


def load_snippets_corpus():
    label_sample_dict = {
        'business': [],
        'computers': [],
        'culture-arts-entertainment': [],
        'education-science': [],
        'engineering': [],
        'health': [],
        'politics-society': [],
        'sports': []
    }
    all_samples = []
    with open('raw_data/Snippets/train.txt', 'r') as input_data:
        for line in input_data.readlines():
            if len(line.strip()) > 0:
                try:
                    all_samples.append(line.strip().lower())
                except:
                    pass
    with open('raw_data/Snippets/test.txt', 'r') as input_data:
        for line in input_data.readlines():
            if len(line.strip()) > 0:
                try:
                    all_samples.append(line.strip().lower())
                except:
                    pass
    all_samples = list(set(all_samples))
    for sample in all_samples:
        label = sample.split()[-1]
        label_sample_dict[label].append(' '.join(sample.split()[:-1]))

    return (label_sample_dict['business'], label_sample_dict['computers'], label_sample_dict['culture-arts-entertainment'],
            label_sample_dict['education-science'], label_sample_dict['engineering'], label_sample_dict['health'],
            label_sample_dict['politics-society'], label_sample_dict['sports'])


if __name__ == '__main__':
    # positive_tweets, negative_tweets = load_twitter_corpus()
    # positive_samples, negative_samples = load_mr_corpus()
    samples = load_snippets_corpus()
    for s in samples:
        print(len(s), s[0])
    """
    print(len(positive_samples), len(negative_samples))
    for idx in range(10):
        print(positive_samples[idx], '###', negative_samples[idx])
    """
