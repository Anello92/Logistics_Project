# Implementation of the Model Deployment Software

# Imports
from flask import Flask, render_template, request, jsonify
import joblib

# App
app = Flask(__name__)

# Load the model and transformers from disk
model_path = 'models/logistic_model.pkl'
transformer_package_type_path = 'models/transformer_package_type.pkl'
transformer_product_type_path = 'models/transformer_product_type.pkl'

model = joblib.load(model_path)
le_package_type = joblib.load(transformer_package_type_path)
le_product_type = joblib.load(transformer_product_type_path)

# Define the main route for the home page and accept only GET requests
@app.route('/', methods=['GET'])
def index():
    # Render the home page using template.html
    return render_template('template.html')

# Define a route to make predictions and accept only POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the 'Weight' value from the submitted form
    weight = int(request.form['Weight'])
    
    # Transform the package type using the previously fitted label encoder
    package_type = le_package_type.transform([request.form['package_type']])[0]
    
    # Use the model to make a prediction based on weight and package type
    prediction = model.predict([[weight, package_type]])[0]
    
    # Convert the encoded prediction back to its original label
    product_type = le_product_type.inverse_transform([prediction])[0]
    
    # Render the home page with the prediction included
    return render_template('template.html', prediction=product_type)

# App
if __name__ == '__main__':
    app.run()
