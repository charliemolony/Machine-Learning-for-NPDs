import pandas as pd
import pickle
from sklearn.metrics import accuracy_score,precision_score,f1_score,roc_auc_score,recall_score,confusion_matrix
# Load the model from the model.pkl file
with open('MLP_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)



# Load the testing data
test_data = pd.read_csv(r"")
#test_data=pd.read_csv("TestingEntireFile.csv",encoding_errors='ignore')
# Vectorize the text using the same TfidfVectorizer as above
X_test = vectorizer.transform(test_data['text'].values.astype('U'))
y_test = test_data['label']

# Make predictions on the test data and compute accuracy
y_pred = model.predict(X_test)
CM=confusion_matrix(y_test,y_pred)

print("Confusion matrix: ")
print(CM)
# accuracy = accuracy_score(y_test, y_pred)
# precision=precision_score(y_test,y_pred)
# f1 = f1_score(y_test, y_pred)
# recall=recall_score(y_test,y_pred)
# AUC=roc_auc_score(y_test,y_pred)
# print("Accuracy: {:.2f}%".format(accuracy*100))
# print("Precision: {:.2f}%".format(precision*100))
# print("F1: {:.2f}%".format(f1*100))
# print("Recall: {:.2f}%".format(recall*100))
# print("AUC: {:.2f}%".format(AUC*100))