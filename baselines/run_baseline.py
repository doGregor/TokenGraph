from baselines_data_preparation import *
from baseline_bert import *


CONFIG = {
    'dataset': 'tag_my_news', # one out of ['twitter', 'mr', 'snippets', 'tag_my_news']
}


samples = load_train_val_test_baseline(CONFIG['dataset'])
train_dataset, val_dataset, test_dataset = samples_to_dataset(samples)

print(train_dataset, val_dataset, test_dataset)


label_dict_dict = {
    'twitter': {'0': 0, '1': 1},
    'mr': {'0': 0, '1': 1},
    'snippets': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7},
    'tag_my_news': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
}

model_dict = {
    'twitter': 'Twitter/twhin-bert-base',
    'mr': 'google-bert/bert-base-uncased',
    'snippets': 'google-bert/bert-base-uncased',
    'tag_my_news': 'google-bert/bert-base-uncased',
}


model_finetuning(train_dataset, val_dataset, test_dataset, 'models/', label_dict_dict[CONFIG['dataset']],
                 model_name=model_dict[CONFIG['dataset']])
