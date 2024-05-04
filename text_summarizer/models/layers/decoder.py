import tensorflow as tf
from models.layers.positionalembedding import PositionalEmbedding
from models.layers.decoderlayers import DecoderLayer

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [ DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, x_attn_mask, context, ctx_attn_mask):
        x_attn_mask = self.pos_embedding.compute_mask(x_attn_mask)
        x_attn_mask = x_attn_mask[:, tf.newaxis, tf.newaxis, :]

        # x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, x_attn_mask, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x