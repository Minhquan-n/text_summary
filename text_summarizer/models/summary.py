import tensorflow as tf

class Summary(tf.Module):
    def __init__(self, tokenizer, transformer):
        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, sentence, max_length=120):
        if len(sentence) == 0:
            sentence = sentence[tf.newaxis]

        token = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=256, return_tensors='tf')
        encoder_attn_mask = token.attention_mask
        encoder_input = token.input_ids

        start_end_token = self.tokenizer('', return_tensors='tf')
        start_end = start_end_token.input_ids
        start_end_attention_mask = start_end_token.attention_mask

        start = tf.cast([start_end[0][0]], tf.int64)
        end = tf.cast(start_end[0][1], tf.int64)

        output_attention_mask = tf.cast([start_end_attention_mask[0][0]], tf.int64)

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        output_attn_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_attn_array = output_attn_array.write(0, output_attention_mask)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            output_attn_mask = tf.transpose(output_attn_array.stack())
            predictions = self.transformer((encoder_input, encoder_attn_mask, output, output_attn_mask), training=False)

            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)

            output_array = output_array.write(i+1, predicted_id[0])
            output_attn_array = output_attn_array.write(i+1, output_attention_mask)

            if predicted_id[0][0] == end:
                break

        output = tf.transpose(output_array.stack())
        output_attention_mask = tf.transpose(output_attn_array.stack())

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text