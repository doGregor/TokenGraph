import sys

from load_datasets import *
from feature_extraction import *
from graph_structure import *
from utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import shuffle
from torch_geometric.data import DataLoader
from nn import *


CONFIG = {
    'dataset': 'snippets', # one out of ['twitter', 'mr', 'snippets']
    'skip_data_generation': False,
    'train_eval_samples_per_class': 20,
    'shuffle_train': True,
    'batch_size': 8,
    'hidden_dim': 128,
    'eval_best': True
}


# (1) generate graphs
if not CONFIG['skip_data_generation']:
    model_dict = {
        'twitter': 'Twitter/twhin-bert-base',
        'mr': 'google-bert/bert-base-uncased',
        'snippets': 'google-bert/bert-base-uncased'
    }
    bert_tokenizer = AutoTokenizer.from_pretrained(model_dict[CONFIG['dataset']])
    bert_model = AutoModel.from_pretrained(model_dict[CONFIG['dataset']])

    existing_graphs = get_saved_graphs(CONFIG['dataset'])

    if CONFIG['dataset'] == 'twitter':
        positive_tweets, negative_tweets = load_twitter_corpus()
        for idx, tweet in enumerate(tqdm(positive_tweets)):
            if 'pos_' + str(idx) + '.pickle' in existing_graphs:
                continue
            graph = generate_graph_from_text(tweet, label=1, tokenizer=bert_tokenizer, llm=bert_model)
            save_graph(graph, file_name=f'pos_{idx}', dataset=CONFIG['dataset'])
        for idx, tweet in enumerate(tqdm(negative_tweets)):
            if 'neg_' + str(idx) + '.pickle' in existing_graphs:
                continue
            graph = generate_graph_from_text(tweet, label=0, tokenizer=bert_tokenizer, llm=bert_model)
            save_graph(graph, file_name=f'neg_{idx}', dataset=CONFIG['dataset'])
    elif CONFIG['dataset'] == 'mr':
        positive_samples, negative_samples = load_mr_corpus()
        for idx, sample in enumerate(tqdm(positive_samples)):
            if 'pos_' + str(idx) + '.pickle' in existing_graphs:
                continue
            graph = generate_graph_from_text(sample, label=1, tokenizer=bert_tokenizer, llm=bert_model)
            save_graph(graph, file_name=f'pos_{idx}', dataset=CONFIG['dataset'])
        for idx, sample in enumerate(tqdm(negative_samples)):
            if 'neg_' + str(idx) + '.pickle' in existing_graphs:
                continue
            graph = generate_graph_from_text(sample, label=0, tokenizer=bert_tokenizer, llm=bert_model)
            save_graph(graph, file_name=f'neg_{idx}', dataset=CONFIG['dataset'])
    elif CONFIG['dataset'] == 'snippets':
        samples = load_snippets_corpus()
        for cls, class_samples in enumerate(samples):
            for idx, sample in enumerate(tqdm(class_samples)):
                if str(cls) + '_' + str(idx) + '.pickle' in existing_graphs:
                    continue
                graph = generate_graph_from_text(sample, label=cls, tokenizer=bert_tokenizer, llm=bert_model)
                save_graph(graph, file_name=f'{cls}_{idx}', dataset=CONFIG['dataset'])

sys.exit(0)

# (2) load data
train_graphs, eval_graph, test_graphs = load_train_eval_test(
    dataset=CONFIG['dataset'],
    num_per_class_train=CONFIG['train_eval_samples_per_class'],
    num_per_class_eval=CONFIG['train_eval_samples_per_class']
)
print(len(train_graphs), len(eval_graph), len(test_graphs))

train_loader = DataLoader(train_graphs, batch_size=CONFIG['batch_size'], shuffle=CONFIG['shuffle_train'])
eval_loader = DataLoader(eval_graph, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=CONFIG['batch_size'], shuffle=False)


# (3) prepare model
model = GAT(hidden_channels=CONFIG['hidden_dim'], out_channels=2, use_hypergraph=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
criterion = torch.nn.CrossEntropyLoss()

acc, precision, recall, f1 = train_eval_model(model=model, train_loader=train_loader, eval_loader=eval_loader,
                                              test_loader=test_loader, loss_fct=criterion, optimizer=optimizer,
                                              num_epochs=200, verbose=1, eval_best=CONFIG['eval_best'])

