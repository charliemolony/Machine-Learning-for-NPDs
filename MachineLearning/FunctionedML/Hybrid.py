##########Do not run Locally############
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.metrics import classification_report
import pickle

print("No import errors")
# Load the data into a Pandas DataFrame
train_data = pd.read_csv("train_data.csv")
test_data=pd.read_csv("test_data.csv")
##train proportion as a decimal
train_proportion = 1

# Calculate the number of samples to use for training
num_train_samples = int(train_proportion * len(train_data))
train_data=train_data.sample(n=num_train_samples,random_state=42)
train_data=train_data.reset_index(drop=True)


# Separate features and target variable
XDB = train_data.drop(columns=['label'])
YDB = train_data['label']
print("data loaded in")

# ##randomly sample the csv file
# X_train=XDB.sample(n=num_train_samples,random_state=42)
# y_train=YDB[XDB.index.isin(X_train.index)]
# # Split the data into training and testing sets

X_train=XDB
y_train=YDB

# X_train = XDB[:num_train_samples]
# y_train = YDB[:num_train_samples]
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

#changing the tuple size from (num,1) to (num,)
X_train=X_train['text'].squeeze()
X_test=X_test['text'].squeeze()

# Convert the text data into a numerical representation using the TF-IDF approach
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("At neural net")

# Train a neural network on the training data
mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=500, activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

print("Completed neural Net")
# Extract the features learned by the neural network
X_train_nn = mlp.predict_proba(X_train)
X_test_nn = mlp.predict_proba(X_test)

# Train an XGBoost model on the extracted features
dtrain = xgb.DMatrix(X_train_nn, label=y_train)
dtest = xgb.DMatrix(X_test_nn, label=y_test)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)


# Evaluate the combined model on the test set
y_pred = xgb_model.predict(dtest)
y_pred_binary = np.round(y_pred)



# Calculate and print various classification metrics
accuracy = accuracy_score(y_test, y_pred_binary)
aoc=roc_auc_score(y_test,y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
cm=confusion_matrix(y_test,y_pred_binary)





print("Neural Net & XGBoost results: ")
print("----------------------------------")
print("training proportion: "+str(train_proportion))
print('Test set accuracy: '+ str(accuracy))
print('Test set precision: '+ str(precision))
print('Test set recall: '+str(recall))
print('Test set F1 score: '+str(f1))
print('Confusion Matrix')
print(cm)


pickle.dump(xgb, open('NN_XGB_model.pkl', 'wb'))