from flask import Flask, render_template, request
import pickle
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model
model = pickle.load(open('ckd.pkl', 'rb'))

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/accuracy", methods=['GET'])
def accuracy():
    return render_template('accuracy.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            bp = request.form['bp']
            al = request.form['al']
            su = request.form['su']
            rbc = request.form['rbc']
            pc = request.form['pc']
            bgr = request.form['bgr']
            sc = float(request.form['sc'])
            pot = float(request.form['pot'])
            hemo = float(request.form['hemo'])
            htn = request.form['htn']
            appet = request.form['appet']
            pe = request.form['pe']

            # Create input array for prediction
            arr = np.array([[bp, al, su, rbc, pc, bgr, sc, pot, hemo, htn, appet, pe]])
            
            # Make a prediction
            pred = model.predict(arr)

            # Render the result
            return render_template('predict.html', prediction=pred)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return "An error occurred. Please check your inputs."

if __name__ == "__main__":
    app.run(debug=True)
