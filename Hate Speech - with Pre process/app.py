from flask import Flask
from flask import Flask, Response, render_template, request, jsonify
from keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import re
app = Flask(__name__)

@app.route("/",methods=['GET', 'POST'])
def hello_world():

    return render_template('home.html')

def load_model(text):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    #tokenizer = Tokenizer()
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    xx = np.array(text)
    xx = tokenizer.texts_to_sequences(xx)
    xx = pad_sequences(xx,maxlen=40)
    print((loaded_model.predict(xx) >= 0.5).astype(int))
    print("Loaded model from disk")
    print(loaded_model.predict(xx))
    return str((loaded_model.predict(xx) >= 0.5).astype(int))


@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    print(text)
    #text = request.data['karim']
    text = clean(text)
    d = {'text': [text]}
    good = '[[1]]'
    hate = '[[0]]'
    X_df = d['text']
    print(X_df)
    y_pred_max = load_model(X_df)
    print(y_pred_max)
    if y_pred_max == good:
        result = "real"
    else:
        result = "fake"
    return jsonify({"result": result})

def clean(text):
    #remove RT if its a retweet
    text = re.sub('(\s*)RT(\s*)','',text)
    #remove repeated words
    text = re.sub("(.)\\1{2,}", "\\1", text)
    #remove numbers 
    text = re.sub(" \d+", " ", text)
    #caractere speciaux
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    
    
    #lower case
    text = text.lower()
    return text