import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
import networkx as nx


def visualize_graph(graph_data, node_size=6, line_width=5):
    graph_viz = to_networkx(graph_data)
    plt.figure(1, figsize=(7, 7))
    nx.draw(graph_viz, cmap=plt.get_cmap('Set1'), arrowstyle='-', node_size=node_size, linewidths=line_width)
    plt.show()
