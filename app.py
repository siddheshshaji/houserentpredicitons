import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import logging
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="https://88e95e8b49ee4033994914ddd7b37c65@o401877.ingest.sentry.io/5262180",
    integrations=[FlaskIntegration()]
)

app = Flask(__name__)

model = pickle.load(open('house_rent_randomforest_1.pkl', 'rb'))
model2 = pickle.load(open('house_rent_1.pkl', 'rb'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [0] * 327
    features = list([x for x in request.form.values()])
    featurevalues = features[0:10]
    feature_values = features[10:]

    featurevalues = [int(s) for s in featurevalues if s.isdigit()]
    
    for x in featurevalues:
        int_features[x]=1
    
    int_features[0] = int(feature_values[0])
    int_features[1] = int(feature_values[1])
    int_features[2] = float(feature_values[2])

    prediction = model.predict([int_features])
    prediction2 = model2.predict([int_features])
    avgprediction = (prediction + prediction2)/2
    output = round(avgprediction[0], 2)

    return render_template('home.html', prediction_text='The estimated rent is ${}'.format(output))

if __name__ == "__main__":
    app.run()