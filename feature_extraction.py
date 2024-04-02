from transformers import AutoTokenizer, AutoModel
import torch


def tokenize_text(text, tokenizer):
    return tokenizer(text, return_tensors="pt")


def decode_tokenized_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids)


def encode_tokens(tokenized_output, bert_model):
    with torch.no_grad():
        return bert_model(**tokenized_output)
