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
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.metrics import classification_report
from pathlib import Path
train_data = pd.read_csv(r"C:\Users\charl\Desktop\Final year Project\Machine Learning\functionedML\training_condensed.csv")
test_data =pd.read_csv(r"C:\Users\charl\Desktop\Final year Project\Machine Learning\functionedML\testing_condensed.csv")

print("No import errors")
# # Load the data into a Pandas DataFrame
# train_data = pd.read_csv("train_data.csv")
# test_data=pd.read_csv("test_data.csv")


##train proportion as a decimal
train_proportion = 1
total_iterations=0
total_accuracy=0
total_precision=0
total_f1=0
total_aoc=0
total_recall=0
num_train_samples = int(train_proportion * len(train_data))
for i in range(1,4):
    print("reinitiliased loop, interation number : "+str(i))

    random_state=42+i
    # Calculate the number of samples to use for training
    
    train_data=train_data.sample(n=num_train_samples,random_state=random_state)
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
    X_train = vectorizer.fit_transform(X_train.values.astype('U'))
    X_test = vectorizer.transform(X_test.values.astype('U'))

    print("At neural net")

    # Train a neural network on the training data
    mlp = MLPClassifier(hidden_layer_sizes=(150, 75,50), max_iter=500, activation='relu', solver='adam', random_state=42)
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


    total_iterations+=1
    # Calculate and print various classification metrics
    total_accuracy+=accuracy_score(y_test, y_pred_binary)
    total_aoc +=roc_auc_score(y_test,y_pred_binary)
    total_precision += precision_score(y_test, y_pred_binary)
    total_recall += recall_score(y_test, y_pred_binary)
    total_f1 =+ f1_score(y_test, y_pred_binary)



print("Neural Net & XGBoost results: ")
print("----------------------------------")
print("training proportion: "+str(num_train_samples))
print('Test set accuracy: '+ str(total_accuracy/total_iterations*100))
print('Test set precision: '+ str(total_precision/total_iterations*100))
print('Test set recall: '+str(total_recall/total_iterations*100))
print('Test set AOC: '+str(total_aoc/total_iterations*100))
print('Test set F1 score: '+str(total_f1/total_iterations*100))



# # Determine the number of text snippets in the CSV file
# train_num_snippets = train_data.shape[0]

# # Count the number of occurrences of each label
# train_label_counts = train_data['label'].value_counts()

# print("Training: Total number of text snippets: ", train_num_snippets)
# print("Training: Label counts:\n", train_label_counts)

# test_num_snippets=test_data.shape[0]
# test_label_counts=test_data['label'].value_counts()

# print("Testing: Total number of text snippets", test_num_snippets)
# print("Testing: Label counts:\n",test_label_counts)