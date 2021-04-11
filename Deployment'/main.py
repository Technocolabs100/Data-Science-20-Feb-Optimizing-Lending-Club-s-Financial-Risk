import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open('Loan.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,4)

    prediction = model.predict(final_features)


    if int(prediction) == 0:
        output = "No"
        return render_template('index.html', prediction_text='Sorry, Your Loan Application is getting Rejected by Lending Club.')
    else:
        output = "Yes"
        return render_template('index.html', prediction_text='Congradulations, Lending Club is accepting your loan application.')


if __name__ == "__main__":
    app.run(debug=True)
    
