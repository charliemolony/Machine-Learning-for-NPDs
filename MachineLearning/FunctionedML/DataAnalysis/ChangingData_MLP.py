import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,roc_auc_score,recall_score


train_data = pd.read_csv(r'C:\Users\charl\Desktop\Final year Project\Machine Learning\functionedML\training_condensed.csv')
test_data =pd.read_csv(r"C:\Users\charl\Desktop\Final year Project\Machine Learning\functionedML\testing_condensed.csv")

print("No import errors")
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
    # Vectorize the text using TfidfVectorizer

    # Train a neural network classifier
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    # print("Made it to training")
    clf.fit(X_train, y_train)
    # print("Made it past training")


    # Vectorize the text using the same TfidfVectorizer as above

    y_test = test_data['label']

# Make predictions on the test data and compute accuracy
    y_pred = clf.predict(X_test)
    total_iterations+=1
    # Calculate and print various classification metrics
    total_accuracy+=accuracy_score(y_test, y_pred)
    total_aoc +=roc_auc_score(y_test,y_pred)
    total_precision += precision_score(y_test, y_pred)
    total_recall += recall_score(y_test, y_pred)
    total_f1 =+ f1_score(y_test, y_pred)



print("Neural Net: ")
print("----------------------------------")
print("training samples: "+str(num_train_samples))
print('Test set accuracy: '+ str(total_accuracy/total_iterations*100))
print('Test set precision: '+ str(total_precision/total_iterations*100))
print('Test set recall: '+str(total_recall/total_iterations*100))
print('Test set AOC: '+str(total_aoc/total_iterations*100))
print('Test set F1 score: '+str(total_f1/total_iterations*100))


