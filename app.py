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
    temp_row = pd.read_excel('sample_1.xlsx')
    features = list([(x) for x in request.form.values()])
    featurevalues = features[0:10]
    feature_values = features[10:]

    try:
        for i in featurevalues:
            temp_row.iat[0,int(i)]=1
    except:
        pass
    
    temp_row.at[0, 'sqfeet'] = feature_values[0]
    temp_row.at[0, 'beds'] = feature_values[1]
    temp_row.at[0, 'baths'] = float(feature_values[2])

    prediction = model.predict(temp_row)
    prediction2 = model2.predict(temp_row)
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