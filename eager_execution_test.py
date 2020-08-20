import tensorflow as tf

(mnist_images, mnist_labels),_ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32), 
    tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000, reshuffle_each_iteration = True).batch(32)


#Build a model
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3,3], activation='relu', input_shape = (None, None, 1)),
    tf.keras.layers.Conv2D(16, [3,3], activation= 'relu'),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(10)
])


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history =[]

def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training = True)

        tf.debugging.assert_equal(logits.shape, (32,10))

        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return loss_history[-1]


def train(epochs):
    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            loss_epoch = train_step(images, labels)
        print(f'Epoch {epoch}: {loss_epoch}')
train(10)

'''
import tensorflow as tf
import os

class Linear(tf.keras.Model):
      def __init__(self):
          super(Linear, self).__init__()
          self.W = tf.Variable(5., name = 'weight', trainable= True)
          self.B = tf.Variable(1., name = 'bias', trainable = True)
      def __call__(self, inputs):
          return inputs*self.W + self.B


Num_egs = 2000
train_inputs = tf.random.normal([Num_egs])
noise = tf.random.normal([Num_egs])
train_outputs = train_inputs*3 + 2 + noise


def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.trainable_variables), loss_value


model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate= 0.01)

print('Initial loss = ',loss(model, train_inputs, train_outputs))

steps = 400
loss_history = []

for i in range(steps):
    grads, loss_value = grad(model, train_inputs, train_outputs)
    loss_history.append(loss_value)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if i%40 == 0:
        print(f'At step {i//40}, loss = {loss_value}')

    
'''
