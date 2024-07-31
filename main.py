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
import warnings
warnings.filterwarnings("ignore")


CONFIG = {
    'dataset': 'twitter', # one out of ['twitter', 'mr', 'snippets', 'tag_my_news']
    'skip_data_generation': True,
    'train_eval_samples_per_class': 20,
    'shuffle_train': True,
    'batch_size': 8,
    'hidden_dim': 128,
    'epochs': 200,
    'eval_best': True,
    'num_runs': 5,
    'verbose': 0,
    'random_samples': False, # False = use same training samples, True = shuffle data and select random training samples
    'num_attention_heads': 1,
    'n_hop_neighborhood': 2, # this has to be set n_hops + 1, e.g. for 2-hop graph = 3
    'gnn_type': 'GAT' # GAT or SAGE or GCN
}


# (1) generate graphs
if not CONFIG['skip_data_generation']:
    model_dict = {
        'twitter': 'Twitter/twhin-bert-base',
        'mr': 'google-bert/bert-base-uncased',
        'snippets': 'google-bert/bert-base-uncased',
        'tag_my_news': 'google-bert/bert-base-uncased',
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
    elif CONFIG['dataset'] == 'tag_my_news':
        samples = load_tagmynews_corpus()
        for cls, class_samples in enumerate(samples):
            for idx, sample in enumerate(tqdm(class_samples)):
                if str(cls) + '_' + str(idx) + '.pickle' in existing_graphs:
                    continue
                graph = generate_graph_from_text(sample, label=cls, tokenizer=bert_tokenizer, llm=bert_model)
                save_graph(graph, file_name=f'{cls}_{idx}', dataset=CONFIG['dataset'])


acc_runs = []
precision_runs = []
recall_runs = []
f1_runs = []
for i in range(CONFIG['num_runs']):
    # (2) load data
    train_graphs, eval_graph, test_graphs = load_train_eval_test(
        dataset=CONFIG['dataset'],
        num_per_class_train=CONFIG['train_eval_samples_per_class'],
        num_per_class_eval=CONFIG['train_eval_samples_per_class'],
        n_hop_neighborhood=CONFIG['n_hop_neighborhood']
    )
    print(len(train_graphs), len(eval_graph), len(test_graphs))

    train_loader = DataLoader(train_graphs, batch_size=CONFIG['batch_size'], shuffle=CONFIG['shuffle_train'])
    eval_loader = DataLoader(eval_graph, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=CONFIG['batch_size'], shuffle=False)

    # (3) prepare model
    num_classes_dict = {
        'twitter': 2,
        'mr': 2,
        'snippets': 8,
        'tag_my_news': 7,
    }
    if CONFIG['gnn_type'] == 'GAT':
        model = GAT(hidden_channels=CONFIG['hidden_dim'], out_channels=num_classes_dict[CONFIG['dataset']],
                    num_attention_heads=CONFIG['num_attention_heads'], use_hypergraph=False)
    elif CONFIG['gnn_type'] == 'GCN':
        model = GCN(hidden_channels=CONFIG['hidden_dim'], out_channels=num_classes_dict[CONFIG['dataset']])
    elif CONFIG['gnn_type'] == 'SAGE':
        model = SAGE(hidden_channels=CONFIG['hidden_dim'], out_channels=num_classes_dict[CONFIG['dataset']])
    else:
        raise 'no valid GNN selected, must be one out of [GAT, GCN, SAGE]'

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    criterion = torch.nn.CrossEntropyLoss()

    acc, precision, recall, f1 = train_eval_model(model=model, train_loader=train_loader, eval_loader=eval_loader,
                                                  test_loader=test_loader, loss_fct=criterion, optimizer=optimizer,
                                                  num_epochs=CONFIG['epochs'], verbose=CONFIG['verbose'],
                                                  eval_best=CONFIG['eval_best'])
    acc_runs.append(acc)
    precision_runs.append(precision)
    recall_runs.append(recall)
    f1_runs.append(f1)

print('Accuracy:', acc_runs, sum(acc_runs)/CONFIG['num_runs'])
print('Precision:', precision_runs, sum(precision_runs)/CONFIG['num_runs'])
print('Recall:', recall_runs, sum(recall_runs)/CONFIG['num_runs'])
print('F1:', f1_runs, sum(f1_runs)/CONFIG['num_runs'])
