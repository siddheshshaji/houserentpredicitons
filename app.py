# IMPORTING THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import logging
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# INITIALIZING SENTRY(FOR LOGGING)
sentry_sdk.init(
    dsn="https://88e95e8b49ee4033994914ddd7b37c65@o401877.ingest.sentry.io/5262180",
    integrations=[FlaskIntegration()]
)

# INITIALIZING THE APP
app = Flask(__name__)

# IMPORTING THE MODELS
random_forest_model = pickle.load(open('house_rent_randomforest_1.pkl', 'rb'))
d_tree_model = pickle.load(open('house_rent_1.pkl', 'rb'))

# 404 PAGE HANDLER
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

# LANDING PAGE
@app.route('/')
def home():
    return render_template('home.html')

# PREDICTION ROUTE
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [0] * 327            # THE MODEL NEEDS AN INPUT OF AN ARRAY OF LENGTH 327 (SINCE THERE ARE 327 FEATURES)
    features = list([x for x in request.form.values()])     # COLLECTING THE INPUT AFTER THE USER PRESSES THE PREDICT BUTTON
    dropdown_values = features[0:10]   # region, cats, dogs, type, furnished, smoking, wheelchair, electric vehicle, laundary options and parking options drop down values
    text_values = features[10:]      # AREA, NO. OF BEDS AND NO. OF BATHS VALUES    

    dropdown_values = [int(s) for s in dropdown_values if s.isdigit()]      # THE isdigit() CONDITION IS USED BECAUSE THE VALUE "NO" IN THE DROP DOWN MENUS ARE NOT ASSIGNED ANY NUMERICAL VALUES
    
    for x in dropdown_values:
        int_features[x]=1              # SETTING CORRESPONDING INDICES (AS VALUES RECEIVED FROM THE DROP DOWN INPUT VALUES, SET IN THE HTML INPUT FORM RESPONSE) AS 1 
    
    int_features[0] = int(text_values[0])   # SETTING CORRESPONDING INDICES (AS VALUES RECEIVED FROM THE TEXT INPUT VALUES, SET IN THE HTML INPUT FORM RESPONSE) AS THE CORRESPONDING VALUE
    int_features[1] = int(text_values[1])   #SAME AS ABOVE
    int_features[2] = float(text_values[2]) #SAME AS ABOVE

    rf_prediction = random_forest_model.predict([int_features])     # RANDOM FOREST MODEL PREDICTION
    dt_prediction2 = d_tree_model.predict([int_features])           # DECISION TREE MODEL PREDICTION
    avgprediction = (rf_prediction + dt_prediction2)/2              # AVERAGE OF BOTH THE MODELS
    output = round(avgprediction[0], 2)                             # ROUNDED TO 2 DECIMAL PLACES

    return render_template('home.html', prediction_text='The estimated rent is ${}'.format(output))

if __name__ == "__main__":
    app.run()
