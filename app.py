from keras.models import load_model
from flask import Flask, jsonify, request
import unicodedata, re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import cPickle
import numpy as np
app = Flask(__name__)

model = load_model("weights.hdf5")

with open('note_v.p', 'rb') as f:
    note_v = cPickle.load(f)

with open('title_v.p', 'rb') as f:
    title_v = cPickle.load(f)

with open('full_categorys.p', 'rb') as f:
    full_categorys = cPickle.load(f)

stemmer = SnowballStemmer('russian')
stopwords = set(stopwords.words('russian'))
pattern = re.compile(ur'\b\w\w+\b', re.U)
def stem(s):
    try:
        s = u''.join([c for c in unicodedata.normalize('NFKD', unicode(s.lower())) if not unicodedata.combining(c)])
        return u' '.join(map(stemmer.stem, filter(lambda x: x.isalpha() and x not in stopwords, pattern.findall(s))))
    except:
        return u''

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


def predict(title, text):
    titles = pad_sentences(
        map(lambda x: filter(None, stem(x).split(' ')), title),
        sequence_length=16)
    notes = pad_sentences(map(lambda x: filter(None, stem(x).split(' ')), text),
                          sequence_length=128)

    x_t = np.array([[title_v[word] if word in title_v else title_v["<UNK/>"] for
                     word in sentence]
                    for sentence in titles])

    x_n = np.array([[note_v[word] if word in note_v else note_v["<UNK/>"] for
                     word in sentence]
                    for sentence in notes])

    return full_categorys[model.predict([x_t, x_n]).argmax()]

@app.route('/predict/', methods=['GET'])
def prediction():
    text = request.args.get('text')

    return jsonify({'doctor_id': predict([u''], [text])})

if __name__ == '__main__':
    app.run(host='0.0.0.0')