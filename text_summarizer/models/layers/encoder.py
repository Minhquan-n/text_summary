import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, phobert_model, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.phobert_model = phobert_model

    def call(self, input_ids, attention_mask):
        outputs = self.phobert_model(input_ids, attention_mask=attention_mask, training=False)

        return outputs.last_hidden_state