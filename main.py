from load_datasets import *
from feature_extraction import *
from graph_structure import *
from utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import shuffle
from torch_geometric.data import DataLoader
from nn import *


# (1) generate graphs
'''
existing_graphs = get_saved_graphs()
positive_tweets, negative_tweets = load_twitter_corpus()

for idx, tweet in enumerate(tqdm(positive_tweets)):
    if 'pos_' + str(idx) + '.pickle' in existing_graphs:
        continue
    graph = generate_graph_from_text(tweet, label=1)
    save_graph(graph, file_name=f'pos_{idx}')

for idx, tweet in enumerate(tqdm(negative_tweets)):
    if 'neg_' + str(idx) + '.pickle' in existing_graphs:
        continue
    graph = generate_graph_from_text(tweet, label=0)
    save_graph(graph, file_name=f'neg_{idx}')
'''


# (2) load data
all_files = [load_graph(x) for x in get_saved_graphs() if x.endswith('pickle')]
all_files = [x for x in all_files if x['y'].shape[0] == 1]
shuffle(all_files)
X_train, X_test = train_test_split(all_files, shuffle=False, test_size=0.2)
print(len(X_train), len(X_test))

train_loader = DataLoader(X_train, batch_size=16, shuffle=True)
test_loader = DataLoader(X_test, batch_size=16, shuffle=False)


# (3) prepare model
model = GAT(hidden_channels=64, out_channels=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
criterion = torch.nn.CrossEntropyLoss()

acc, precision, recall, f1 = train_eval_model(model=model, train_loader=train_loader, test_loader=test_loader,
                                              loss_fct=criterion, optimizer=optimizer, num_epochs=100,
                                              verbose=1)

