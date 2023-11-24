from flask import Flask, render_template, flash, request
from zxcvbn import zxcvbn
import numpy as np
import pickle
import re
import math

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/main/', methods=['GET', 'POST'])
def mainpage():
    if request.method == "POST":
        enteredPassword = request.form['password']
    else:
        return render_template('index.html')
    
    value=zxcvbn(enteredPassword)
    Password = [enteredPassword]

    check = 2
    #Checking in top 10 million most commonly used password 
    with open('10-million-password-list.txt', 'r') as file:
            for line in file:
                if enteredPassword in line:
                    check = 0

    print(check)
                    
    # Calculate Entropy
    count = 0;
    if bool(re.search(r'[A-Z]', enteredPassword)) : 
        count += 26
    if bool(re.search(r'[a-z]', enteredPassword)) : 
        count += 26
    if bool(re.search(r'\d', enteredPassword)) :
        count += 10
    if bool(re.search(r'[!@#$%^&*()_+={}[\]|\\:;"\'<>,.?/]', enteredPassword)) :
        count += 32

    entropy = len(enteredPassword) * np.log2(count)   
    print(count)
    print(entropy) 

    with open('DecisionTree_Model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('LogisticRegression_Model.pkl', 'rb') as file:
        loaded_model1 = pickle.load(file)
    with open('NaiveBayes_Model.pkl', 'rb') as file:
        loaded_model2 = pickle.load(file)
    with open('NeuralNetwork_Model.pkl', 'rb') as file:
        loaded_model3 = pickle.load(file)
    with open('RandomForest_Model.pkl', 'rb') as file:
        loaded_model4 = pickle.load(file)

    # Use the loaded model for predictions   
    # Predict the strength
    DecisionTree_Test = loaded_model.predict(Password)
    LogisticRegression_Test = loaded_model1.predict(Password)
    NaiveBayes_Test = loaded_model2.predict(Password)
    RandomForest_Test = loaded_model4.predict(Password)
    NeuralNetwork_Test = loaded_model3.predict(Password)
    avg = (np.array(int(value['score'])) + DecisionTree_Test + LogisticRegression_Test + NaiveBayes_Test + RandomForest_Test + NeuralNetwork_Test) / 6

    avg_test = np.array([avg])
    return render_template("main.html", DecisionTree=DecisionTree_Test,
                                        LogReg=LogisticRegression_Test,
                                        NaiveBayes=NaiveBayes_Test,
                                        RandomForest=RandomForest_Test,
                                        NeuralNetwork=NeuralNetwork_Test,
                                        Average=avg_test,
                                        )

if __name__ == "__main__":
    app.run(debug=True)
