import tensorflow as tf
import numpy as np
import functools
from tensorflow.keras.layers import Dense, Flatten, Dropout
#from sklearn.model_selection import train_test_split

LABEL_COLUMN = 'chd'
LABELS = [0, 1]
SELECT_COLUMNS = ['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age','chd']
FEATURE_COLUMNS = SELECT_COLUMNS[:-1]

file_path = 'heart.csv'

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=462, # Max 462
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
  return dataset

raw_train_data = get_dataset(file_path,select_columns=SELECT_COLUMNS)
raw_test_data = get_dataset(file_path,select_columns=SELECT_COLUMNS)

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

NUMERIC_FEATURES = ['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))


numeric_column = tf.feature_column.numeric_column('numeric', shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

CATEGORIES = {'famhist': ['Present', 'Absent']}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

model = tf.keras.Sequential([
  preprocessing_layer,
  Dense(256, activation='relu'),
  Dropout(0.2),
  Dense(128, activation='relu'),
  Dropout(0.2),
  Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=100)

test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

print("Reached EOF")
