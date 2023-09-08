import torch
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertForSequenceClassification,AutoConfig
from sklearn.model_selection import train_test_split
import evaluate
import tensorflow as tf
import evaluate


accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
precision = evaluate.load('precision')
recall = evaluate.load('recall')

def compute_metrics(eval_pred):
    print("compute metrics initialised")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)['accuracy']
    f1_score = f1.compute(predictions=predictions, references=labels)['f1']
    precision_score = precision.compute(predictions=predictions, references=labels)['precision']
    recall_score = recall.compute(predictions=predictions, references=labels)['recall']
    print(f"accuracy: {accuracy_score}, f1: {f1_score}, precision: {precision_score}, recall: {recall_score}")

    return {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    }

###microsoft/codebert-base different type of transfer learning project 

tokenizer = AutoTokenizer.from_pretrained('neulab/codebert-c')
#model= BertForSequenceClassification.from_pretrained('neulab/codebert-c')
# Count total number of parameters in the CodeBERT model



# # Load the configuration
config_path = "./results/checkpoint-2500/config.json"
config = AutoConfig.from_pretrained(config_path)

# Load the model weights
weights_path = "./results/checkpoint-2500/pytorch_model.bin"
model = BertForSequenceClassification.from_pretrained(weights_path, config=config)





# Freeze all parameters except the last layer
for name, param in model.named_parameters():
    if 'classifier' in name:  # Train only the last layer
        param.requires_grad = True
        
    else:
        param.requires_grad = False


# Load the training dataset
train_df = pd.read_csv("training_condensed.csv")

train_df = train_df.dropna()
train_texts = train_df["text"].tolist()
train_encodings = tokenizer(train_texts, truncation=True, padding=True)



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
    num_train_epochs=50,              # total number of training epochs
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
    compute_metrics=compute_metrics,
)

print("made it to training")
# Fine-tune the model on the training dataset
trainer.train()
print("made it past training")
# Load the test dataset
test_df = pd.read_csv("testing_condensed.csv")

test_df = test_df.dropna()
test_texts= test_df["text"].tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True)




# Convert the test labels to tensors
test_actual_labels = torch.tensor(test_df['label'].tolist())

# Initialize the test dataset
test_dataset = [{"input_ids": test_encodings['input_ids'][i],
                  "attention_mask": test_encodings['attention_mask'][i],
                  "label": test_actual_labels[i]} for i in range(len(test_df))]


# Evaluate the model on the test dataset
eval_output = trainer.evaluate(test_dataset)


