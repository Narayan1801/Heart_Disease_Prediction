import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=True)

app = Flask(__name__)
model = pickle.load(open('heart_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    

    return render_template('main.html', prediction_text='The Chances of Having Heart Disease is {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trough request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

    
    
    
