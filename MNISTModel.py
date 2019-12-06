import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

input_shape = (28, 28)
print("--Get data--")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("--Process data--")
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

print("--Make model--")
model = tf.keras.models.Sequential([
  Conv2D(32, (3,3), input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(64, (3,3), input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(input_shape=(28, 28)),
  Dense(10, activation='sigmoid')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=10, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

#Save Model
print("--Save model--")
model.save('MNIST.h5')
print("Model saved as MNIST.h5")
