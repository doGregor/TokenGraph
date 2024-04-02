import pickle
import os
import sys
from random import shuffle


def save_graph(graph, file_name, dataset):
    with open(f'graphs/graphs_{dataset}/{str(file_name)}.pickle', 'wb') as handle:
        pickle.dump({'graph': graph}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(graph_name, dataset):
    if 'pickle' in graph_name:
        graph_name = graph_name.split('.')[0]
    with open(f'graphs/graphs_{dataset}/{graph_name}.pickle', 'rb') as handle:
        graph = pickle.load(handle)
    return graph['graph']


def get_saved_graphs(dataset):
    return os.listdir('graphs/graphs_' + dataset)


def load_train_eval_test(dataset, num_per_class_train=20, num_per_class_eval=20):
    if dataset == 'twitter':
        pos_samples = []
        neg_samples = []
        for graph_name in get_saved_graphs(dataset=dataset):
            graph = load_graph(graph_name, dataset)
            if graph['y'].shape[0] != 1:
                continue
            if graph_name[:3] == 'neg':
                neg_samples.append(graph)
            if graph_name[:3] == 'pos':
                pos_samples.append(graph)
        shuffle(pos_samples)
        shuffle(neg_samples)
        train_graphs = pos_samples[:num_per_class_train] + neg_samples[:num_per_class_train]
        eval_graphs = (pos_samples[num_per_class_train:num_per_class_train+num_per_class_eval] +
                       neg_samples[num_per_class_train:num_per_class_train+num_per_class_eval])
        test_graphs = (pos_samples[num_per_class_train+num_per_class_eval:] +
                       neg_samples[num_per_class_train+num_per_class_eval:])
        shuffle(train_graphs)
        shuffle(eval_graphs)
        shuffle(test_graphs)
        return train_graphs, eval_graphs, test_graphs


'''
if __name__ == '__main__':
    all_files = [load_graph(x) for x in get_saved_graphs() if x.endswith('pickle')]
    for g in all_files:
        if g['y'].shape[0] != 1:
            print(g['y'])
'''
