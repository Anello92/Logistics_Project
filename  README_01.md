### README.md

Machine Learning Model Deployment

This project demonstrates how to build and deploy a Machine Learning model using Python and Flask. It consists of three main files: `deploy.py`, `template.html`, and `model.py`.

Files

1. **deploy.py**: This file contains the Flask application code to deploy the Machine Learning model.
2. **template.html**: The HTML template for the web interface where users can input data and get predictions.
3. **model.py**: This file is responsible for building, training, and saving the Machine Learning model.

Setup Instructions

Prerequisites

- Python 3.x
- Flask
- Joblib
- Scikit-learn
- Pandas

Installation

1. Clone the repository:

   
   git clone https://github.com/yourusername/ml-model-deployment.git
   cd ml-model-deployment
   
2. Install the required packages:

   pip install flask joblib scikit-learn pandas
  

Running the Project

1. Train the Model**:

   Run `model.py` to train and save the model and transformers.

   
   python model.py


2. Start the Flask Application**:

   Run `deploy.py` to start the web server.


   python deploy.py
   

3. Access the Web Application**:

   Open your web browser and go to `http://127.0.0.1:5000`. You will see the web interface where you can input data for predictions.


File Details

deploy.py


from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('models/logistic_model.pkl')
le_package_type = joblib.load('models/transformer_package_type.pkl')
le_product_type = joblib.load('models/transformer_product_type.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('template.html')

@app.route('/predict', methods=['POST'])
def predict():
    weight = int(request.form['Weight'])
    package_type = le_package_type.transform([request.form['package_type']])[0]
    prediction = model.predict([[weight, package_type]])[0]
    product_type = le_product_type.inverse_transform([prediction])[0]
    return render_template('template.html', prediction=product_type)

if __name__ == '__main__':
app.run()

template.html


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Learning Model Deployment</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f5f5f7;
            color: #333;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background: #fff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 500px;
            text-align: center;
        }
        h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 30px;
        }
        label {
            display: block;
            margin-bottom: 10px;
            font-size: 16px;
            color: #555;
            text-align: left;
        }
        input[type="text"], select {
            width: 100%;
            padding: 12px 20px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-sizing: border-box;
            font-size: 16px;
        }
        input[type="submit"] {
            background-color: #0071e3;
            color: white;
            padding: 14px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        input[type="submit"]:hover {
            background-color: #005bb5;
        }
        .prediction {
            margin-top: 20px;
            padding: 20px;
            background-color: #f0f0f5;
            border: 1px solid #dcdcdc;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Machine Learning Model Deployment</h1>
        <form method="POST" action="/predict">
            <label for="Weight">Package Weight (grams):</label>
            <input type="text" id="Weight" name="Weight" required>
            <label for="package_type">Package Type:</label>
            <select id="package_type" name="package_type" required>
                <option value="Cardboard Box">Cardboard Box</option>
                <option value="Bubble Wrap">Bubble Wrap</option>
            </select>
            <input type="submit" value="Predict">
        </form>
        {% if prediction %}
        <div class="prediction">
            <h2>Prediction: {{ prediction }}</h2>
        </div>
        {% endif %}
    </div>
</body>
</html>


model.py

python
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = {
    'Package_Weight_Gr': [212, 215, 890, 700, 230, 240, 730, 780, 218, 750, 202, 680],
    'Package_Type': ['Cardboard Box', 'Cardboard Box', 'Bubble Wrap', 'Bubble Wrap', 'Cardboard Box', 'Cardboard Box', 'Bubble Wrap', 'Bubble Wrap', 'Cardboard Box', 'Bubble Wrap', 'Cardboard Box', 'Bubble Wrap'],
    'Product_Type': ['Smartphone', 'Tablet', 'Tablet', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Smartphone', 'Tablet']
}

df = pd.DataFrame(data)

X = df[['Package_Weight_Gr', 'Package_Type']]
y = df['Product_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

le_package_type = LabelEncoder()
le_package_type.fit(X_train['Package_Type'])

le_product_type = LabelEncoder()
le_product_type.fit(y_train)

X_train['Package_Type'] = le_package_type.transform(X_train['Package_Type'])
X_test['Package_Type'] = le_package_type.transform(X_test['Package_Type'])

y_train = le_product_type.transform(y_train)
y_test = le_product_type.transform(y_test)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_model = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: ", round(acc_model, 2))

print("\nClassification Report:\n")
report = classification_report(y_test, y_pred)
print(report)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(le_package_type, 'models/transformer_package_type.pkl')
joblib.dump(le_product_type, 'models/transformer_product_type.pkl')


Conclusion

This project demonstrates how to build and deploy a Machine Learning model using Flask and Python. Follow the setup instructions to get the application running on your local machine. Feel free to customize and extend this project to suit your needs. Happy coding!

Instructions for Use

1. Save the above content as `README.md` in your project directory.
2. Ensure all the other files (`deploy.py`, `template.html`, `model.py`) are also saved in the appropriate locations as specified in the `README.md`.
3. Commit all these files to your GitHub repository.

This `README.md` file provides a comprehensive guide to setting up, running, and understanding the project, making it easier for others to use and contribute.