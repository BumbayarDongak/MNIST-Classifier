import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import random
import matplotlib.pyplot as plt

model=tf.keras.models.load_model('fashion_mnist.keras')

(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test=x_test/255

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
             'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
def predict_image(index):
    img=x_test[index]
    img_expanded=img.reshape(1, 28, 28)
    predictions=model.predict(img_expanded)
    predicted_class=predictions[0].argmax()
    plt.imshow(img, cmap='gray')
    plt.title(f'Предсказано:{class_names[predicted_class]}\n'
              f'Истинный класс: {class_names[y_test[index]]}')
    plt.axis('off')
    plt.show()

random_index=random.randint(0,len(x_test)-1)
predict_image(random_index)





