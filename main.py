# from flask import Flask, request, jsonify
# import pickle
# from mofrl_files.ml_model import prediction

# app = Flask("Loan_prediction")

# @app.route('/', methods=['POST'])
# def predict():
#     # return "Loan Prediction Model Application!!"
#     loan = request.get_json()
#     with open('./model')



# if __name__ == "__main__":
#     app.run(debug = True, host = '0.0.0.0', port = 9696)
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
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    if prediction == 1:
        output = "Loan accepted"
    
    else:
        output = "Loan rejected"

    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)