from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from baselines_data_preparation import *


tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base")
model = AutoModelForSequenceClassification.from_pretrained("Twitter/twhin-bert-base",
                                                           num_labels=2)

training_args = TrainingArguments(output_dir="test_trainer",
                                  num_train_epochs=3,
                                  per_device_train_batch_size=16,
                                  evaluation_strategy="epoch")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


positive_tweets, negative_tweets = load_twitter_corpus()
df = samples_to_df(positive_tweets, negative_tweets)
dataset_train, dataset_test = dataset_from_df(df)


def tokenize_function(examples):
    return tokenizer(examples["text"],
                     max_length=128,
                     padding="max_length",
                     truncation=True)


dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_test = dataset_test.map(tokenize_function, batched=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,
)

trainer.train()
