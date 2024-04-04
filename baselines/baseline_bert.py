import json
import logging
from os.path import join as pj
import torch
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def get_metrics():
    metric_accuracy = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    def compute_metric_search(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='micro')

    def compute_metric_all(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'f1': metric_f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
            'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
            'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
        }
    return compute_metric_search, compute_metric_all


def model_finetuning(train_data, eval_data, test_data, output_dir, label2id, seq_length=128, eval_steps=500, model_name='bert-base-cased', random_seed=42):
    print('starting finetuning...')
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(list(id2label.keys()))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
        )
    tokenized_train = train_data.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=seq_length),
        batched=True)
    tokenized_eval = eval_data.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=seq_length),
        batched=True)

    # setup metrics
    compute_metric_search, compute_metric_all = get_metrics()

    # setup trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            seed=random_seed
        ),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metric_search,
        model_init=lambda x: AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            return_dict=True,
            id2label=id2label,
            label2id=label2id
        )
    )

    trainer.train()
    trainer.save_model(pj(output_dir, 'best_model'))
    best_model_path = pj(output_dir, 'best_model')
    
    tokenized_test = test_data.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=seq_length),
        batched=True)

    # testing
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            seed=random_seed
        ),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metric_all,
    )
    summary_file = pj(output_dir, 'metric_summary.json')
    result = {f'test/{k}': v for k, v in trainer.evaluate().items()}
    logging.info(json.dumps(result, indent=4))
    with open(summary_file, 'w') as f:
        json.dump(result, f)
