import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from underthesea import word_tokenize
from models.transformer import Transformer
from models.summary import Summary

def prepare_batch(inp, inp_attn_mask, out, out_attn_mask, label):
    input = inp
    input_attention_mask = inp_attn_mask
    output = out
    output_attention_mask = out_attn_mask
    label = label

    return (input, input_attention_mask, output, output_attention_mask), label

def make_batches(ds):
  return (ds.shuffle(20000).batch(4).map(prepare_batch, tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE))

def transformer_model():

    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    phobert_model = TFAutoModel.from_pretrained('vinai/phobert-base')
    phobert_config = phobert_model.config

    # Khởi tạo các Hyperparameters cho mô hình, các tham số dựa trên config của mô hình phoBERT
    num_layers = 4
    d_model = phobert_config.hidden_size
    dff = phobert_config.intermediate_size
    num_heads = 4
    vocab_size = phobert_config.vocab_size
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        phobert_model = phobert_model,
        dropout_rate=dropout_rate)
    return transformer, tokenizer

def summary_model(transformer, tokenizer):
    text = 'Một trong những vấn_đề lớn nhất của NLP là vấn_đề dữ_liệu .'
    summ = 'Vấn_đề lớn của NLP là dữ_liệu .'

    token = tokenizer(text, return_tensors='tf')
    text_input_ids = token.input_ids
    text_attention_mask = token.attention_mask

    summ_token = tokenizer(summ, return_tensors='tf')
    summ_input_ids = summ_token.input_ids
    summ_attention_mask = summ_token.attention_mask

    out = transformer([text_input_ids, text_attention_mask, summ_input_ids, summ_attention_mask])

    transformer.load_weights('weights/transformer_full.h5')

    summarizer = Summary(tokenizer, transformer)
    return summarizer