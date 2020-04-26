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
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    
    if output == 1:
        string = "detected!"
    else:
        string = "not detected!"
        
    #print(string)
    
    prediction_text='Diabetes ' + string
        
    return render_template('predict.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)