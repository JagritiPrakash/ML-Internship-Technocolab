#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:09:16 2020

@author: jagriti
"""



import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = np.array([[float(x) for x in request.form.values()]])
    pred= model.predict(final_features)  
    if(pred[0]==0):
    	a="Healthy"
    else:
    	a= "Not Healthy"
    return ( "Output is : "+ a )

    

   

if __name__ == "__main__":
    app.run(debug=True)
    
    
  