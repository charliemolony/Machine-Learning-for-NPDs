from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn import svm
from keras.layers import concatenate
from keras.models import Model

#import matplotlib.pyplot as plt


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

print(X_train.shape)
X_train=X_train['text'].squeeze()
X_test=X_test['text'].squeeze()

print(X_train.shape)

# Convert the text data into a numerical representation using the TF-IDF approach
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

X_train = X_train.toarray()
X_test=   X_test.toarray()

# create a neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)


X_train_features = model.predict(X_train)
X_test_features = model.predict(X_test)

# train an SVM classifier on the extracted features
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_features, y_train)

# test the classifier on the testing set
y_pred = svm.predict(X_test_features)

# compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC-ROC Score:", auc_roc)
print("Confusion Matrix:\n", cm)

