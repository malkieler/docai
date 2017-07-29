from keras.models import load_model
from flask import Flask, jsonify, request
import re
import _pickle as cPickle
import numpy as np
app = Flask(__name__)

model = load_model("weights.hdf5")



with open('note_v.p', 'rb') as f:
    note_v = cPickle.load(f)

with open('full_categorys.p', 'rb') as f:
    full_categorys = cPickle.load(f)

pattern = re.compile(r'\b\w\w+\b')

def stem(s):
    return ' '.join(pattern.findall(s))

def pad_sentences(sentences, padding_word="<PAD/>", sequence_length = 30):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i][:sequence_length]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def predict(text):
    notes = pad_sentences(map(lambda x: filter(None, stem(x).split(' ')), text),
                          sequence_length=256)

    x_n = np.array([[note_v[word] if word in note_v else note_v["<UNK/>"] for
                     word in sentence]
                    for sentence in notes])

    pred_values = {full_categorys[idx]: value for idx, value in enumerate(model.predict([x_n])[0])}

    sorted_values = sorted(pred_values.items(), key = lambda x: x[1], reverse=True)
    if sorted_values[0][1] > 0.5:
        return [sorted_values[0][0]]
    elif sorted_values[0][1] + sorted_values[1][1] > 0.5:
        return [sorted_values[0][0], sorted_values[1][0]]
    else:
        return [sorted_values[0][0], sorted_values[1][0], sorted_values[2][0]]

@app.route('/predict/', methods=['GET'])
def prediction():
    text = request.args.get('text')

    return jsonify({'doctor_ids': predict([text])})

context = ('cert.crt', 'key.key')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, ssl_context=context)