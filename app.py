import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template, send_from_directory
import numpy as np
import pandas as pd
import os

app=Flask(__name__)

#Load the model
refmodel=pickle.load(open('rf_model.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=refmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Age Prediction of the Abalone is {}".format(output))


@app.route('/bck.jpg')
def background():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'bck.jpg')

if __name__=="__main__":
    app.run(debug=True)