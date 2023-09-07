import pandas as pd
import pickle

# Load the model from the model.pkl file
with open('MLP_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer from the vectorizer.pkl file
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the analysis data
analysis_data=pd.read_csv(r"")

# Separate features and target variable
X_analyse = analysis_data.drop(columns=['label','url'])
y_analyse = analysis_data['label']
Z_analyse=analysis_data['url']

# Vectorize the text data
X_analyse = vectorizer.transform(X_analyse['text'].squeeze())

# Predict the labels using the model
y_pred = model.predict(X_analyse)

# Create a new DataFrame with the predicted labels and the actual labels
results = pd.DataFrame({'predicted': y_pred, 'actual': y_analyse, 'url': Z_analyse})

# Split the results into 4 data frames: true positive, true negative, false positive, false negative
tp = results[(results['predicted'] == 1) & (results['actual'] == 1)]
tn = results[(results['predicted'] == 0) & (results['actual'] == 0)]
fp = results[(results['predicted'] == 1) & (results['actual'] == 0)]
fn = results[(results['predicted'] == 0) & (results['actual'] == 1)]

tp.to_csv('true_positive.csv', index=False)
tn.to_csv('true_negative.csv', index=False)
fp.to_csv('false_positive.csv',index=False)
fn.to_csv('false_negative.csv',index=False)
# Print the shape of each data frame
print('True positive:', tp.shape)
print('True negative:', tn.shape)
print('False positive:', fp.shape)
print('False negative:', fn.shape)





# # Create empty lists to hold the results for each category
# # Create empty lists to hold the results for each category
# tp = []
# tn = []
# fp = []
# fn = []

# # Loop through the data and classify each row using the model
# for i, row in enumerate(data):
#     label = int(row['label'])
#     # Use the model to predict the label for the current row
#     predicted_label = model.predict(X[i])[0]
#     # Determine which category the current row belongs to and append it to the corresponding list
#     if label == 1 and predicted_label == 1:
#         tp.append(row)
#     elif label == 0 and predicted_label == 0:
#         tn.append(row)
#     elif label == 0 and predicted_label == 1:
#         fp.append(row)
#     elif label == 1 and predicted_label == 0:
#         fn.append(row)

# # Write the results to separate CSV files
# with open('true_positive.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data[0].keys())
#     writer.writeheader()
#     writer.writerows(tp)

# with open('true_negative.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data[0].keys())
#     writer.writeheader()
#     writer.writerows(tn)

# with open('false_positive.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data[0].keys())
#     writer.writeheader()
#     writer.writerows(fp)

# with open('false_negative.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data[0].keys())
#     writer.writeheader()
#     writer.writerows(fn)
