import io
import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd
import sqlite3


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report


batch_size = 100
seed = 123
dir="C:/Users/charl/Desktop/Final year Project/GitHubApi/training/functionedDatabase"
train_ds=tf.keras.utils.text_dataset_from_directory(dir, labels='inferred', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds=tf.keras.utils.text_dataset_from_directory(dir, labels='inferred', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_layer = tf.keras.layers.Embedding(1000, 5)


# result = embedding_layer(tf.constant([1, 2, 3]))
# result.numpy()

result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
result.shape



# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')


# Vocabulary size and number of words in a sequence. ####changed the vocab size from 10,000 50,000
vocab_size = 50000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


embedding_dim=50
##changed the embedding dim from 30 to 50 same with number of neurons
model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(50,activation='relu'),
  Dense(1)
])


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

##(from_logits=True) ####binary cross entropy acc= 53%
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=[tensorboard_callback])

weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('FunctionedVectors.tsv', 'w', encoding='utf-8')
out_m = io.open('FunctionedMetadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()

test_dir=r"C:\Users\charl\Desktop\Final year Project\GitHubApi\testing\functionedDatabase"
test_ds = tf.keras.utils.text_dataset_from_directory(test_dir, labels='inferred', batch_size=3)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Get the true labels of the test dataset
y_true = []
for x, y in test_ds:
  y_true.extend(y.numpy())

# Get the predicted labels of the test dataset
y_pred = model.predict(test_ds)
y_pred = (y_pred > 0.5).astype(int).flatten()

# Calculate precision, recall, and F1 score
print(classification_report(y_true, y_pred))

testing_loss, testing_accuracy = model.evaluate(test_ds)

print("TESTING Loss: ", testing_loss)
print("TESTING Accuracy: ", testing_accuracy)