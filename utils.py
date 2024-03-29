import pickle
import os


def save_graph(graph, file_name):
    with open(f'graphs/{str(file_name)}.pickle', 'wb') as handle:
        pickle.dump({'graph': graph}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(graph_name):
    if 'pickle' in graph_name:
        graph_name = graph_name.split('.')[0]
    with open(f'graphs/{graph_name}.pickle', 'rb') as handle:
        graph = pickle.load(handle)
    return graph['graph']


def get_saved_graphs():
    return os.listdir('graphs')


if __name__ == '__main__':
    all_files = [load_graph(x) for x in get_saved_graphs() if x.endswith('pickle')]
    for g in all_files:
        if g['y'].shape[0] != 1:
            print(g['y'])
