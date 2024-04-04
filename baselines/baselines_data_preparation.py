import sys
from load_datasets import *
import random
from datasets import Dataset
from random import shuffle
import pyarrow as pa


def load_train_val_test_baseline(dataset, num_per_class=20):
    if dataset == 'twitter':
        samples = load_twitter_corpus()
    elif dataset == 'mr':
        samples = load_mr_corpus()
    elif dataset == 'snippets':
        samples = load_snippets_corpus()
    elif dataset == 'tag_my_news':
        samples = load_tagmynews_corpus()
    train_samples = []
    train_labels = []
    val_samples = []
    val_labels = []
    test_samples = []
    test_labels = []
    for class_idx, class_samples in enumerate(samples):
        shuffle(class_samples)
        train_samples += class_samples[:num_per_class]
        train_labels += [class_idx] * num_per_class
        val_samples += class_samples[num_per_class:num_per_class + num_per_class]
        val_labels += [class_idx] * num_per_class
        test_samples += class_samples[num_per_class + num_per_class:]
        test_labels += [class_idx] * (len(class_samples) - 2 * num_per_class)
    c = list(zip(train_samples, train_labels))
    random.shuffle(c)
    train_samples, train_labels = zip(*c)
    c = list(zip(val_samples, val_labels))
    random.shuffle(c)
    val_samples, val_labels = zip(*c)
    c = list(zip(test_samples, test_labels))
    random.shuffle(c)
    test_samples, test_labels = zip(*c)
    return train_samples, train_labels, val_samples, val_labels, test_samples, test_labels


def samples_to_dataset(sample_tuple):
    names = ["text", "label", "label_name"]
    train_table = pa.Table.from_arrays(
        [pa.array(sample_tuple[0]), pa.array(sample_tuple[1]), pa.array(list(map(str, sample_tuple[1])))], names=names)
    train_dataset = Dataset(train_table)
    val_table = pa.Table.from_arrays(
        [pa.array(sample_tuple[2]), pa.array(sample_tuple[3]), pa.array(list(map(str, sample_tuple[3])))], names=names)
    val_dataset = Dataset(val_table)
    test_table = pa.Table.from_arrays(
        [pa.array(sample_tuple[4]), pa.array(sample_tuple[5]), pa.array(list(map(str, sample_tuple[5])))], names=names)
    test_dataset = Dataset(test_table)
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    samples = load_train_val_test_baseline('twitter')
    train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = samples
    print(len(train_samples), len(val_samples), len(test_samples))
    print(train_samples[0], train_labels[0])
    train_dataset, val_dataset, test_dataset = samples_to_dataset(samples)
    print(train_dataset, val_dataset, test_dataset)
