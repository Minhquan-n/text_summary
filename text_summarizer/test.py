from transformers import AutoTokenizer, TFAutoModel
from underthesea import word_tokenize
from utils import *
from models.transformer import Transformer
from models.summary import Summary

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

text2 = 'CEO Tim Cook nói, từ việc hợp tác với các nhà cung cấp địa phương, đến hỗ trợ các dự án cung cấp nước sạch và các cơ hội giáo dục, chúng tôi cam kết sẽ tiếp tục tăng cường các kết nối tại Việt Nam. Tim Cook dành nhiều lời khen khi nói về đất nước, con người Việt Nam. Ông nói không có nơi nào như Việt Nam, một đất nước sôi động và xinh đẹp. Từ khi hoạt động tại Việt Nam đến nay, Apple đang hỗ trợ hơn 200.000 việc làm trực tiếp và gián tiếp, thông qua chuỗi cung ứng và hệ sinh thái iOS. Apple cho biết sẽ tăng cường khoản chi cho các nhà cung cấp tại Việt Nam. Công ty đã chi gần 400.000 tỉ đồng từ năm 2019 thông qua chuỗi cung ứng địa phương và tăng hơn gấp đôi mức chi hằng năm cho Việt Nam trong cùng kỳ.'
text2 = word_tokenize(text2, format='text')
output_text = summarizer(text2)
print(output_text.replace('_', ' '))