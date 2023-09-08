import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("---------------------The last time this code was edidted was 13/03 at 17:49-------------------")
# Load the pre-trained CodeBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Load the pre-trained CodeBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
# for param in model.bert.bert.parameters():
#     param.requires_grad = False
# Load the training dataset
train_df = pd.read_csv("train_data.csv")

# Encode the training dataset using the CodeBERT tokenizer
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)

# Convert the training labels to tensors
train_labels = torch.tensor(train_df['label'].tolist())

# Initialize the training dataset
train_dataset = [{"input_ids": train_encodings['input_ids'][i],
                  "attention_mask": train_encodings['attention_mask'][i],
                  "label": train_labels[i]} for i in range(len(train_df))]

# Split the training dataset into training and validation datasets
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)
print("made it to the Neural Net")
# Fine-tune the model on the training dataset
trainer.train()

# Load the test dataset
test_df = pd.read_csv("test_data.csv")

# Encode the test dataset using the CodeBERT tokenizer
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)

# Convert the test labels to tensors
test_labels = torch.tensor(test_df['label'].tolist())

# Initialize the test dataset
test_dataset = [{"input_ids": test_encodings['input_ids'][i],
                  "attention_mask": test_encodings['attention_mask'][i],
                  "label": test_labels[i]} for i in range(len(test_df))]

# Evaluate the model on the test dataset
eval_output = trainer.evaluate(test_dataset)



# Evaluate the model on the test dataset
eval_output = trainer.evaluate(test_dataset)

# Get the predicted labels
y_pred = eval_output.predictions.argmax(axis=1)

# Calculate the evaluation metrics
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")
