import requests

def get_prediction(text):
    url = 'http://localhost:5000/predict'
    params ={'text': text}
    res = requests.get(url, params=params)
    print res
    return res.json()