import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle

# Load the data into a Pandas DataFrame
train_data = pd.read_csv("Total_training.csv")
test_data=pd.read_csv("Total_testing.csv")

# Separate features and target variable
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']

train_proportion = 1


# Calculate the number of samples to use for training

X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

print(X_train.shape)
X_train=X_train['text'].squeeze()
X_test=X_test['text'].squeeze()

print(X_train.shape)
print(X_test.shape)
# Convert the text data into a numerical representation using the TF-IDF approach
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

pickle.dump(vectorizer, open('Total_Vectors.pkl', 'wb'))


# print(X_train.shape)

# # Train an XGBoost classifier on the training data
# model = XGBClassifier()
# model.fit(X_train, y_train)


# y_pred = model.predict(X_test)

# #print("y prediction size: "+str(+y_pred.size))

# # Compute the evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# auc_roc = roc_auc_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# print("AUC-ROC Score:", auc_roc)
# print("Confusion Matrix:\n", cm)


# pickle.dump(model, open('Total_XGB_model.pkl', 'wb'))




