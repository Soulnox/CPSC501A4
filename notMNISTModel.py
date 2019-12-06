import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0
 
print("--Make model--")
model = tf.keras.models.Sequential([
  Conv2D(32, (3,3), input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(64, (3,3), input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(input_shape=(28, 28)),
  Dropout(0.2),
  Dense(128, activation='relu'),
  Dropout(0.2),
  Dense(10, activation='sigmoid')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=20, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

print("--Save model--")
model.save('notMNIST.h5')
print("Model saved as notMNIST.h5")
