import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
#from tensorflow.keras.layers.experimental.preprocessing import TextVectorization #when runnnung virtually
from tensorflow.keras.layers import TextVectorization # for when running locally
#import tensorflow_datasets as tfds
from sklearn.metrics import classification_report

# Load the data into a Pandas DataFrame
train_data = pd.read_csv("train_data.csv")
test_data=pd.read_csv("test_data.csv")

# Separate features and target variable
XDB = train_data.drop(columns=['label'])
YDB = train_data['label']

train_proportion = 1

# Calculate the number of samples to use for training
num_train_samples = int(train_proportion * len(train_data))

# Split the data into training and testing sets
X_train = XDB[:num_train_samples]
y_train = YDB[:num_train_samples]
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

#changing the tuple size from (num,1) to (num,)
X_train=X_train['text'].squeeze()
X_test=X_test['text'].squeeze()

# Convert the text data into a numerical representation using the TF-IDF approach
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
    'activation': ['relu', 'tanh', 'logistic']
}

# Create an MLPClassifier model with fixed hyperparameters
mlp = MLPClassifier(max_iter=500, solver='adam', random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding mean test score
print('Best hyperparameters:', grid_search.best_params_)
print('Best mean test score:', grid_search.best_score_)