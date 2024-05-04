from flask import Flask, render_template, request
from utils import summary_model, transformer_model
from underthesea import word_tokenize

app = Flask(__name__)
print('Chuẩn bị mô hình...')
transformer, tokenizer = transformer_model()
model = summary_model(transformer=transformer, tokenizer=tokenizer)
print('Mô hình hoàn tất.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['input_text']
    input_text = word_tokenize(input_text, format='text')
    output = model(input_text)
    output_text = output.replace('_', ' ') 

    return output_text

if __name__ == '__main__':
    app.run(debug=True)
