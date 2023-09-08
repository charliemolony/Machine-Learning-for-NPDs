import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,roc_auc_score,recall_score
import pickle
# Load the training data
train_data = pd.read_csv("train_data.csv")

# Vectorize the text using TfidfVectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_data['text'])
y_train = train_data['label']

# Train a neural network classifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
print("Made it to training")
model.fit(X_train, y_train)
print("Made it past training")

# Load the testing data
test_data = pd.read_csv("test_data.csv")

# Vectorize the text using the same TfidfVectorizer as above
X_test = tfidf.transform(test_data['text'])
y_test = test_data['label']

# Make predictions on the test data and compute accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision=precision_score(y_test,y_pred)
recall = recall_score(y_test, y_pred)
AUC=roc_auc_score(y_test,y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))
print("AUC: {:.2f}%".format(AUC*100))

pickle.dump(model, open('MLP_model.pkl', 'wb'))