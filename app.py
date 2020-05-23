import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('house_rent_randomforest_1.pkl', 'rb'))
model2 = pickle.load(open('house_rent_1.pkl', 'rb'))

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

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)