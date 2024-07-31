import pickle
import os
import sys
import random
from random import shuffle
import torch_geometric.transforms as T
import torch


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
    return [x for x in os.listdir('graphs/graphs_' + dataset) if x.endswith('pickle')]


def change_n_hop_neighborhood(graph, n_hops):
    number_of_nodes = graph.x.shape[0]
    edge_index = [[], []]
    for i_from in range(0, number_of_nodes):
        for i_to in range(i_from+1, i_from+n_hops):
            if i_to < number_of_nodes:
                edge_index[0].append(i_from)
                edge_index[1].append(i_to)
    graph.edge_index = torch.LongTensor(edge_index)
    graph = T.RemoveDuplicatedEdges()(graph)
    graph = T.ToUndirected()(graph)
    return graph


def load_train_eval_test(dataset, num_per_class_train=20, num_per_class_eval=20, n_hop_neighborhood=2,
                         random_samples=True, random_seed=100):
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
        if random_samples:
            shuffle(pos_samples)
            shuffle(neg_samples)
        else:
            random.Random(random_seed).shuffle(pos_samples)
            random.Random(random_seed).shuffle(neg_samples)
        train_graphs = pos_samples[:num_per_class_train] + neg_samples[:num_per_class_train]
        eval_graphs = (pos_samples[num_per_class_train:num_per_class_train+num_per_class_eval] +
                       neg_samples[num_per_class_train:num_per_class_train+num_per_class_eval])
        test_graphs = (pos_samples[num_per_class_train+num_per_class_eval:] +
                       neg_samples[num_per_class_train+num_per_class_eval:])
    elif dataset == 'mr':
        pos_samples = []
        neg_samples = []
        for graph_name in get_saved_graphs(dataset=dataset):
            graph = load_graph(graph_name, dataset)
            if graph['y'].shape[0] != 1 or graph['x'].shape[0] < 3:
                continue
            if graph_name[:3] == 'neg':
                neg_samples.append(graph)
            if graph_name[:3] == 'pos':
                pos_samples.append(graph)
        if random_samples:
            shuffle(pos_samples)
            shuffle(neg_samples)
        else:
            random.Random(random_seed).shuffle(pos_samples)
            random.Random(random_seed).shuffle(neg_samples)
        train_graphs = pos_samples[:num_per_class_train] + neg_samples[:num_per_class_train]
        eval_graphs = (pos_samples[num_per_class_train:num_per_class_train + num_per_class_eval] +
                       neg_samples[num_per_class_train:num_per_class_train + num_per_class_eval])
        test_graphs = (pos_samples[num_per_class_train + num_per_class_eval:] +
                       neg_samples[num_per_class_train + num_per_class_eval:])
    elif dataset == 'snippets':
        sample_dict = {
            '0': [],
            '1': [],
            '2': [],
            '3': [],
            '4': [],
            '5': [],
            '6': [],
            '7': [],
        }
        for graph_name in get_saved_graphs(dataset=dataset):
            graph = load_graph(graph_name, dataset)
            if graph['y'].shape[0] != 1 or graph['x'].shape[0] < 3:
                continue
            sample_dict[graph_name[0]].append(graph)
        train_graphs = []
        eval_graphs = []
        test_graphs = []
        for label in list(sample_dict.keys()):
            if random_samples:
                shuffle(sample_dict[label])
            else:
                random.Random(random_seed).shuffle(sample_dict[label])
            train_graphs += sample_dict[label][:num_per_class_train]
            eval_graphs += sample_dict[label][num_per_class_train:num_per_class_train + num_per_class_eval]
            test_graphs += sample_dict[label][num_per_class_train + num_per_class_eval:]
    elif dataset == 'tag_my_news':
        sample_dict = {
            '0': [],
            '1': [],
            '2': [],
            '3': [],
            '4': [],
            '5': [],
            '6': [],
        }
        for graph_name in get_saved_graphs(dataset=dataset):
            graph = load_graph(graph_name, dataset)
            if graph['y'].shape[0] != 1 or graph['x'].shape[0] < 3:
                continue
            sample_dict[graph_name[0]].append(graph)
        train_graphs = []
        eval_graphs = []
        test_graphs = []
        for label in list(sample_dict.keys()):
            if random_samples:
                shuffle(sample_dict[label])
            else:
                random.Random(random_seed).shuffle(sample_dict[label])
            train_graphs += sample_dict[label][:num_per_class_train]
            eval_graphs += sample_dict[label][num_per_class_train:num_per_class_train + num_per_class_eval]
            test_graphs += sample_dict[label][num_per_class_train + num_per_class_eval:]

    if n_hop_neighborhood != 3:
        train_graphs = [change_n_hop_neighborhood(g, n_hop_neighborhood) for g in train_graphs]
        eval_graphs = [change_n_hop_neighborhood(g, n_hop_neighborhood) for g in eval_graphs]
        test_graphs = [change_n_hop_neighborhood(g, n_hop_neighborhood) for g in test_graphs]

    if random_samples:
        shuffle(train_graphs)
        shuffle(eval_graphs)
        shuffle(test_graphs)
    else:
        random.Random(random_seed).shuffle(train_graphs)
        random.Random(random_seed).shuffle(eval_graphs)
        random.Random(random_seed).shuffle(test_graphs)
    return train_graphs, eval_graphs, test_graphs


if __name__ == '__main__':
    all_files = [load_graph(x, 'mr') for x in get_saved_graphs('mr') if x.endswith('pickle')]
    print(len(all_files))
    for g in all_files[:10]:
        print(g)
        if g['y'].shape[0] != 1:
            print(g)
