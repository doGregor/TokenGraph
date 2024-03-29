from transformers import AutoTokenizer, AutoModel
import torch


BERT_TWITTER_TOKENIZER = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
BERT_TWITTER_MODEL = AutoModel.from_pretrained('Twitter/twhin-bert-base')


def tokenize_text(text):
    return BERT_TWITTER_TOKENIZER(text, return_tensors="pt")


def decode_tokenized_text(token_ids):
    return BERT_TWITTER_TOKENIZER.decode(token_ids)


def encode_tokens(tokenized_output):
    with torch.no_grad():
        return BERT_TWITTER_MODEL(**tokenized_output)


if __name__ == '__main__':
    sample = '#FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)'
    print(len(BERT_TWITTER_TOKENIZER.tokenize(sample)))
    tk = BERT_TWITTER_TOKENIZER(sample, return_tensors="pt")
    print(tk)
    print(tk['input_ids'][0].shape)
    model_out = encode_tokens(tk)
    print(model_out.last_hidden_state)
