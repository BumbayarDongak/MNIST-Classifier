import numpy as np
import tensorflow as tf

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return max(0, x)

# inputs=[8, 18, 73, 2, 10]
# weights=[-0.5, 0.7, 0.5, 0.4, 0.3]
# bias=-0.8

# weighted_inputs=[inputs[i]*weights[i] for i in range(len(inputs))]
# result=sum(weighted_inputs)+bias

# activated_output=relu(result)
# print('рейтинг игрока:', activated_output)

cheetos=tf.constant([2, 8, 3, 7, 8], dtype=tf.float32)
lays=tf.constant([1, 7, 3, 6, 5], dtype=tf.float32)
weights=tf.constant([0.3, 0.4, 0.5, 0.4, 0.5], dtype=tf.float32)

benefit1=tf.multiply(cheetos, weights)
benefit2=tf.multiply(lays, weights)
result1=tf.reduce_sum(benefit1)
result2=tf.reduce_sum(benefit2)

concl1=tf.nn.relu(result1)
concl2=tf.nn.relu(result2)

tf.print('Рейтинг привлекательности Cheetos:', tf.round(concl1))
tf.print('Рейтинг привлекательности Lays:', tf.round(concl2))
