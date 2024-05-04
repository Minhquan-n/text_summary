from transformers import AutoTokenizer, TFAutoModel, PhobertTokenizer
from underthesea import word_tokenize
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from utils import *
from models.transformer import Transformer
from models.schedule import CustomSchedule, masked_accuracy, masked_loss

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
phobert_model = TFAutoModel.from_pretrained('vinai/phobert-base')
phobert_config = phobert_model.config

phobert_model.trainable = False

# Khởi tạo các danh sách lưu các đoạn văn bản
article = []
article_attention_masks = []
summaries = []
summaries_attention_masks = []
labels = []
MAX_LENGTH=100

# Khởi tạo các Hyperparameters cho mô hình, các tham số dựa trên config của mô hình phoBERT
num_layers = 6
d_model = phobert_config.hidden_size
dff = phobert_config.intermediate_size
num_heads = 4
vocab_size = phobert_config.vocab_size
dropout_rate = 0.1

dataset = pd.read_csv('data/phobert_summary.csv')
ds_article = dataset['article'].to_list()
ds_summaries = dataset['summary'].to_list()

for i in range(len(dataset)):
    art = tokenizer(ds_article[i], return_tensors='tf', max_length=256, padding='max_length', truncation=True)
    art_input_ids = art.input_ids.numpy()[0]
    art_attn_mask = art.attention_mask.numpy()[0]
    article.append(art_input_ids)
    article_attention_masks.append(art_attn_mask)

    summ = tokenizer(ds_summaries[i], return_tensors='tf', max_length=MAX_LENGTH, padding='max_length', truncation=True)
    summ_input_ids = summ.input_ids.numpy()[0]
    summ_attn_mask = summ.attention_mask.numpy()[0][:-1]
    summaries.append(summ_input_ids[:-1])
    summaries_attention_masks.append(summ_attn_mask)
    labels.append(summ_input_ids[1:])

dataset = tf.data.Dataset.from_tensor_slices((article, article_attention_masks, summaries, summaries_attention_masks, labels))
datasets = make_batches(dataset)

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    phobert_model = phobert_model,
    dropout_rate=dropout_rate)

text = 'Một trong những vấn_đề lớn nhất của NLP là vấn_đề dữ_liệu .'
summ = 'Vấn_đề lớn của NLP là dữ_liệu .'

# token = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors='tf')
token = tokenizer(text, return_tensors='tf')
text_input_ids = token.input_ids
text_attention_mask = token.attention_mask

# summ_token = tokenizer(summ, padding='max_length', max_length=256, truncation=True, return_tensors='tf')
summ_token = tokenizer(summ, return_tensors='tf')
summ_input_ids = summ_token.input_ids
summ_attention_mask = summ_token.attention_mask

out = transformer([text_input_ids, text_attention_mask, summ_input_ids, summ_attention_mask])

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
transformer.summary()
# transformer.fit(datasets, epochs=10)

