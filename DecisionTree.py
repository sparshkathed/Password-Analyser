# Import the necessary Libraries
import pandas as pd

# For text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# For creating a pipeline
from sklearn.pipeline import Pipeline

# Classifier Model (Decision Tree)
from sklearn.tree import DecisionTreeClassifier

# To save the trained model on local storage
import pickle

# Import tqdm for progress bar
from tqdm import tqdm

# Read the File
data = pd.read_csv('training.csv')

# Features which are passwords
features = data.values[:, 1].astype('str')

# Labels which are the strength of password
labels = data.values[:, -1].astype('int')

# Sequentially apply a list of transforms and a final estimator
classifier_model = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char')),
    ('decisionTree', DecisionTreeClassifier()),
])

# Create a tqdm progress bar
with tqdm(total=len(features)) as pbar:
    for i, (feature, label) in enumerate(zip(features, labels)):
        classifier_model.fit([feature], [label])
        pbar.update(1)

# Training Accuracy
training_accuracy = classifier_model.score(features, labels)
print('Training Accuracy: ', training_accuracy)

model_bytes = pickle.dumps(classifier_model)

# Write the serialized model to a file
with open('DecisionTree_Model.pkl', 'wb') as file:
    file.write(model_bytes)