import tensorflow as tf
from models.layers.encoder import Encoder
from models.layers.decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                input_vocab_size, target_vocab_size, phobert_model, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(phobert_model)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=target_vocab_size, dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation='relu')

    def call(self, inputs):

        context, ctx_attn_mask, x, x_attn_mask  = inputs

        context = self.encoder(context, ctx_attn_mask)

        x = self.decoder(x, x_attn_mask, context, ctx_attn_mask)

        # Final layer
        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output
        return logits