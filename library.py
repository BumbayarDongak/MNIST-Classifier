import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
weight=tf.constant([5, 10, 20, 30, 45, 50], dtype=tf.float32)
distance=tf.constant([50, 100, 200, 300, 450, 500], dtype=tf.float32)
x_train=tf.stack([weight, distance], axis=1)
fuel=tf.constant([500, 1000, 2000, 3000, 4500, 5000], dtype=tf.float32)

model=keras.Sequential([
    layers.Dense(200, activation='relu', input_shape=(2,)),
    layers.Dense(100),
    layers.Dense(1),
    ])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(x_train, fuel, epochs=1000, verbose=0)
print('Gone')

w_test=tf.constant([8, 12, 18], dtype=tf.float32)
d_test=tf.constant([60, 120, 180], dtype=tf.float32)
x_test=tf.stack([w_test, d_test], axis=1)
predictions=model.predict(x_test)
print('входные данные:', x_test)
print('прогнозы:', predictions.flatten())