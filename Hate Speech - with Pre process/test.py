
from flask import Flask
from flask import Flask, Response, render_template, request, jsonify
from keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle

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
xx = ["fuck you bitch pussy"]
xx = tokenizer.texts_to_sequences(xx)
xx = pad_sequences(xx,maxlen=40)
print((loaded_model.predict(xx) >= 0.5).astype(int))
print("Loaded model from disk")
print(loaded_model.predict(xx))