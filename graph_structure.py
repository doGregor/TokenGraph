from torch_geometric.data import HeteroData, Data
from feature_extraction import tokenize_text, encode_tokens
import torch
import torch_geometric.transforms as T


def generate_graph_from_text(text_sample, label, context_window=3):
    tokenized_sample = tokenize_text(text_sample)
    token_vectors = encode_tokens(tokenized_sample)

    node_ids = tokenized_sample['input_ids'][0].tolist()[1:-1]
    node_features = token_vectors.last_hidden_state[0, 1:-1, :]

    edge_index = [[], []]
    for i_from in range(0, len(node_ids)):
        for i_to in range(i_from+1, i_from+context_window):
            if i_to < len(node_ids):
                edge_index[0].append(i_from)
                edge_index[1].append(i_to)
    # remove duplicates and make undirected

    hyperedge_index = [[], []]
    node_ids = torch.LongTensor(node_ids)
    unique_elements, counts = node_ids.unique(return_counts=True)
    at_least_twice = unique_elements[counts >= 2]
    for idx, val in enumerate(at_least_twice.tolist()):
        val_pos = torch.nonzero(node_ids == val).squeeze()
        for index_val_pos in val_pos:
            hyperedge_index[0].append(index_val_pos.item())
            hyperedge_index[1].append(idx)

    graph = Data(x=node_features,
                 edge_index=torch.LongTensor(edge_index),
                 y=torch.LongTensor([label]),
                 hyperedge_index=torch.LongTensor(hyperedge_index))

    graph = T.RemoveDuplicatedEdges()(graph)
    graph = T.ToUndirected()(graph)

    return graph
