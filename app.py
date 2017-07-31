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

stopwords = ['\u0438',
 '\u0432',
 '\u0432\u043e',
 '\u043d\u0435',
 '\u0447\u0442\u043e',
 '\u043e\u043d',
 '\u043d\u0430',
 '\u044f',
 '\u0441',
 '\u0441\u043e',
 '\u043a\u0430\u043a',
 '\u0430',
 '\u0442\u043e',
 '\u0432\u0441\u0435',
 '\u043e\u043d\u0430',
 '\u0442\u0430\u043a',
 '\u0435\u0433\u043e',
 '\u043d\u043e',
 '\u0434\u0430',
 '\u0442\u044b',
 '\u043a',
 '\u0443',
 '\u0436\u0435',
 '\u0432\u044b',
 '\u0437\u0430',
 '\u0431\u044b',
 '\u043f\u043e',
 '\u0442\u043e\u043b\u044c\u043a\u043e',
 '\u0435\u0435',
 '\u043c\u043d\u0435',
 '\u0431\u044b\u043b\u043e',
 '\u0432\u043e\u0442',
 '\u043e\u0442',
 '\u043c\u0435\u043d\u044f',
 '\u0435\u0449\u0435',
 '\u043d\u0435\u0442',
 '\u043e',
 '\u0438\u0437',
 '\u0435\u043c\u0443',
 '\u0442\u0435\u043f\u0435\u0440\u044c',
 '\u043a\u043e\u0433\u0434\u0430',
 '\u0434\u0430\u0436\u0435',
 '\u043d\u0443',
 '\u0432\u0434\u0440\u0443\u0433',
 '\u043b\u0438',
 '\u0435\u0441\u043b\u0438',
 '\u0443\u0436\u0435',
 '\u0438\u043b\u0438',
 '\u043d\u0438',
 '\u0431\u044b\u0442\u044c',
 '\u0431\u044b\u043b',
 '\u043d\u0435\u0433\u043e',
 '\u0434\u043e',
 '\u0432\u0430\u0441',
 '\u043d\u0438\u0431\u0443\u0434\u044c',
 '\u043e\u043f\u044f\u0442\u044c',
 '\u0443\u0436',
 '\u0432\u0430\u043c',
 '\u0432\u0435\u0434\u044c',
 '\u0442\u0430\u043c',
 '\u043f\u043e\u0442\u043e\u043c',
 '\u0441\u0435\u0431\u044f',
 '\u043d\u0438\u0447\u0435\u0433\u043e',
 '\u0435\u0439',
 '\u043c\u043e\u0436\u0435\u0442',
 '\u043e\u043d\u0438',
 '\u0442\u0443\u0442',
 '\u0433\u0434\u0435',
 '\u0435\u0441\u0442\u044c',
 '\u043d\u0430\u0434\u043e',
 '\u043d\u0435\u0439',
 '\u0434\u043b\u044f',
 '\u043c\u044b',
 '\u0442\u0435\u0431\u044f',
 '\u0438\u0445',
 '\u0447\u0435\u043c',
 '\u0431\u044b\u043b\u0430',
 '\u0441\u0430\u043c',
 '\u0447\u0442\u043e\u0431',
 '\u0431\u0435\u0437',
 '\u0431\u0443\u0434\u0442\u043e',
 '\u0447\u0435\u0433\u043e',
 '\u0440\u0430\u0437',
 '\u0442\u043e\u0436\u0435',
 '\u0441\u0435\u0431\u0435',
 '\u043f\u043e\u0434',
 '\u0431\u0443\u0434\u0435\u0442',
 '\u0436',
 '\u0442\u043e\u0433\u0434\u0430',
 '\u043a\u0442\u043e',
 '\u044d\u0442\u043e\u0442',
 '\u0442\u043e\u0433\u043e',
 '\u043f\u043e\u0442\u043e\u043c\u0443',
 '\u044d\u0442\u043e\u0433\u043e',
 '\u043a\u0430\u043a\u043e\u0439',
 '\u0441\u043e\u0432\u0441\u0435\u043c',
 '\u043d\u0438\u043c',
 '\u0437\u0434\u0435\u0441\u044c',
 '\u044d\u0442\u043e\u043c',
 '\u043e\u0434\u0438\u043d',
 '\u043f\u043e\u0447\u0442\u0438',
 '\u043c\u043e\u0439',
 '\u0442\u0435\u043c',
 '\u0447\u0442\u043e\u0431\u044b',
 '\u043d\u0435\u0435',
 '\u0441\u0435\u0439\u0447\u0430\u0441',
 '\u0431\u044b\u043b\u0438',
 '\u043a\u0443\u0434\u0430',
 '\u0437\u0430\u0447\u0435\u043c',
 '\u0432\u0441\u0435\u0445',
 '\u043d\u0438\u043a\u043e\u0433\u0434\u0430',
 '\u043c\u043e\u0436\u043d\u043e',
 '\u043f\u0440\u0438',
 '\u043d\u0430\u043a\u043e\u043d\u0435\u0446',
 '\u0434\u0432\u0430',
 '\u043e\u0431',
 '\u0434\u0440\u0443\u0433\u043e\u0439',
 '\u0445\u043e\u0442\u044c',
 '\u043f\u043e\u0441\u043b\u0435',
 '\u043d\u0430\u0434',
 '\u0431\u043e\u043b\u044c\u0448\u0435',
 '\u0442\u043e\u0442',
 '\u0447\u0435\u0440\u0435\u0437',
 '\u044d\u0442\u0438',
 '\u043d\u0430\u0441',
 '\u043f\u0440\u043e',
 '\u0432\u0441\u0435\u0433\u043e',
 '\u043d\u0438\u0445',
 '\u043a\u0430\u043a\u0430\u044f',
 '\u043c\u043d\u043e\u0433\u043e',
 '\u0440\u0430\u0437\u0432\u0435',
 '\u0442\u0440\u0438',
 '\u044d\u0442\u0443',
 '\u043c\u043e\u044f',
 '\u0432\u043f\u0440\u043e\u0447\u0435\u043c',
 '\u0445\u043e\u0440\u043e\u0448\u043e',
 '\u0441\u0432\u043e\u044e',
 '\u044d\u0442\u043e\u0439',
 '\u043f\u0435\u0440\u0435\u0434',
 '\u0438\u043d\u043e\u0433\u0434\u0430',
 '\u043b\u0443\u0447\u0448\u0435',
 '\u0447\u0443\u0442\u044c',
 '\u0442\u043e\u043c',
 '\u043d\u0435\u043b\u044c\u0437\u044f',
 '\u0442\u0430\u043a\u043e\u0439',
 '\u0438\u043c',
 '\u0431\u043e\u043b\u0435\u0435',
 '\u0432\u0441\u0435\u0433\u0434\u0430',
 '\u043a\u043e\u043d\u0435\u0447\u043d\u043e',
 '\u0432\u0441\u044e',
 '\u043c\u0435\u0436\u0434\u0443']

class Porter:
    PERFECTIVEGROUND = re.compile(
        "((ив|ивши|ившись|ыв|ывши|ывшись)|((?<=[ая])(в|вши|вшись)))$")
    REFLEXIVE = re.compile("(с[яь])$")
    ADJECTIVE = re.compile(
        "(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$")
    PARTICIPLE = re.compile("((ивш|ывш|ующ)|((?<=[ая])(ем|нн|вш|ющ|щ)))$")
    VERB = re.compile(
        "((ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)|((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)))$")
    NOUN = re.compile(
        "(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$")
    RVRE = re.compile("^(.*?[аеиоуыэюя])(.*)$")
    DERIVATIONAL = re.compile(".*[^аеиоуыэюя]+[аеиоуыэюя].*ость?$")
    DER = re.compile("ость?$")
    SUPERLATIVE = re.compile("(ейше|ейш)$")
    I = re.compile("и$")
    P = re.compile("ь$")
    NN = re.compile("нн$")

    def stem(word):
        word = word.lower()
        word = word.replace('ё', 'е')
        m = re.match(Porter.RVRE, word)
        if m and m.groups():
            pre = m.group(1)
            rv = m.group(2)
            temp = Porter.PERFECTIVEGROUND.sub('', rv, 1)
            if temp == rv:
                rv = Porter.REFLEXIVE.sub('', rv, 1)
                temp = Porter.ADJECTIVE.sub('', rv, 1)
                if temp != rv:
                    rv = temp
                    rv = Porter.PARTICIPLE.sub('', rv, 1)
                else:
                    temp = Porter.VERB.sub('', rv, 1)
                    if temp == rv:
                        rv = Porter.NOUN.sub('', rv, 1)
                    else:
                        rv = temp
            else:
                rv = temp

            rv = Porter.I.sub('', rv, 1)

            if re.match(Porter.DERIVATIONAL, rv):
                rv = Porter.DER.sub('', rv, 1)

            temp = Porter.P.sub('', rv, 1)
            if temp == rv:
                rv = Porter.SUPERLATIVE.sub('', rv, 1)
                rv = Porter.NN.sub('н', rv, 1)
            else:
                rv = temp
            word = pre + rv
        return word

    stem = staticmethod(stem)

pattern = re.compile(r'\b\w\w+\b', re.U)

def stem(s):
    s = s.lower()
    s = re.sub(r'\d', '', s)
    return u' '.join(map(Porter.stem,
                         filter(lambda x: x.isalpha() and x not in stopwords,
                                pattern.findall(s))))


def pad_sentences(sentences, padding_word="<PAD/>", sequence_length = 30):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    print(sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i][:sequence_length]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def predict(text):
    notes = pad_sentences(list(map(lambda x: stem(x).split(' '), text)),
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