'''
Flask Application to run model
'''
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle 

app = Flask(__name__)
model = pickle.load(open('svc-model.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    '''
    Rendering results from model to HTML
    '''
    input = [x for x in request.form.values()]
    app.logger.info(f"Input : {input[0]}")
    text = pd.Series(input[0])
    text_vector = tfidf.transform(text)    
    output = model.predict(text_vector)
    app.logger.info(f"Output : {output[0]}")

    return render_template('index.html',input_text=input[0],predicted_text=output[0])

if __name__=="__main__":
    app.run(debug=True)