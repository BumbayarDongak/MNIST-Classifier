from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train=x_train/255.0
x_test=x_test/255.0

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Одежда: {y_train[i]}")
    plt.axis('off')
plt.show()
model=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(15, activation='softmax',)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history=model.fit(
    x_train, y_train,
    epochs=10, batch_size=32,
    validation_data=(x_test, y_test)
)

test_loss, test_accuracy=model.evaluate(x_test, y_test)
print(f"ТОЧНОСТЬ МОДЕЛИ: {test_accuracy:.2f}")
model.save('fashion_mnist.keras')