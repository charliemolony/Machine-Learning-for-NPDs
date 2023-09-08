import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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

    X_train=XDB
    y_train=YDB

    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']

    # print(X_train.shape)
    X_train=X_train['text'].squeeze()
    X_test=X_test['text'].squeeze()

    # print(X_train.shape)

    # Convert the text data into a numerical representation using the TF-IDF approach
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train.values.astype('U'))
    X_test = vectorizer.transform(X_test.values.astype('U'))

    # print(X_train.shape)

    # Train an XGBoost classifier on the training data
    model = XGBClassifier()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    #print("y prediction size: "+str(+y_pred.size))

    # Compute the evaluation metrics

    total_accuracy += accuracy_score(y_test, y_pred)
    total_precision += precision_score(y_test, y_pred)
    total_recall += recall_score(y_test, y_pred)

    total_aoc += roc_auc_score(y_test, y_pred)
    total_iterations+=1


print("XGBoost results: ")
print("Training Samples: ",num_train_samples)
print("----------------------------------")

print("Accuracy:", total_accuracy/total_iterations*100)
print("Precision:", total_precision/total_iterations*100)
print("Recall:", total_recall/total_iterations*100)
print("AUC-ROC Score:", total_aoc/total_iterations*100)





