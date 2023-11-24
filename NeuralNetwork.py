# Import the necessary Libraries
import pandas as pd

# For text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# For creating a pipeline
from sklearn.pipeline import Pipeline

# Classifier Model (MultiLayer Perceptron)
from sklearn.neural_network import MLPClassifier

# To save the trained model on local storage
# from sklearn.externals import joblib
import pickle

# Read the File
data = pd.read_csv('training.csv')
# data=data[0:10]

# Features which are passwords
features = data.values[:, 1].astype('str')

# Labels which are strength of password
labels = data.values[:, -1].astype('int')

# Sequentially apply a list of transforms and a final estimator
classifier_model = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char')),
                ('mlpClassifier', MLPClassifier(solver='adam', 
                                                alpha=1e-5, 
                                                max_iter=400,
                                                activation='logistic')),
])

# Fit the Model
classifier_model.fit(features, labels)

# Training Accuracy
print('Training Accuracy: ',classifier_model.score(features, labels))
model_bytes = pickle.dumps(classifier_model)

# Save model for Logistic Regression
with open('NeuralNetwork_Model.pkl', 'wb') as file:
    file.write(model_bytes)