import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.pipeline import Pipeline

# Load the model from the model.pkl file


# Load the model from the model.pkl file
with open('MLP_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

file_path = r""  # Update with your file path

with open(file_path, 'r') as f:
    text_data = f.read()  # Read the entire file as a single string

# Create a DataFrame with the text data
df = pd.DataFrame({'text': [text_data]})
print(df)

X_test = df['text']
print(X_test)
X_test = vectorizer.transform(X_test)
# Fit the pipeline on the text data

pred = model.predict(X_test)

# Print the predictions
print(pred)
# Create a pipeline with vectorizer and model

# Fit the pipeline on the text data


# Create a confusion matrix to get true positives, true negatives, false positives, false negatives
# cm = confusion_matrix(test_df[target_column], y_pred)
# print(cm)
# # Extract indices of true positive, true negative, false positive, false negative cases
# tp_indices = (y_pred == test_df[target_column]) & (y_pred == 1)
# tn_indices = (y_pred == test_df[target_column]) & (y_pred == 0)
# fp_indices = (y_pred != test_df[target_column]) & (y_pred == 1)
# fn_indices = (y_pred != test_df[target_column]) & (y_pred == 0)

# # Create four separate dataframes for true positive, true negative, false positive, false negative cases
# tp_df = test_df[tp_indices]
# tn_df = test_df[tn_indices]
# fp_df = test_df[fp_indices]
# fn_df = test_df[fn_indices]

# # Save dataframes to CSV files
# tp_df.to_csv('true_positive.csv', index=False)
# tn_df.to_csv('true_negative.csv', index=False)
# fp_df.to_csv('false_positive.csv', index=False)
# fn_df.to_csv('false_negative.csv', index=False)