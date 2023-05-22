import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow.keras.layers as tfkl

# Loading MNIST
(trainX, trainY) , (testX, testY) = mnist.load_data()

# Normalizing data
trainX = np.divide(trainX, 255.0)
testX = np.divide(testX, 255.0)

# 12000 samples for validation
valX = trainX[-12000:]
valY = trainY[-12000:]
# Remaining for training
trainX = trainX[:-12000]
trainY = trainY[:-12000]

# LSTM architecture
# 21386 parameters
input = tfkl.Input(shape=(28,28))
rnncell = tfkl.LSTM(128)(input)
output = tfkl.Dense(10)(rnncell)

model = tf.keras.Model(inputs=input, outputs=output)

# Print out model architecture
model.summary()

# Compile
# Using Adam optimizer - standard more optimized than SGD
# SparseCategoricalCrossEntropy - Loss fxn for multi-class classification where labels are integer valued and not encoded
# From logits as we don't have a mapping at the output
# Report accuracy as a metric
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Tensorboard callback for later visualization
tb = tf.keras.callbacks.TensorBoard(log_dir='lstm_logs')


start = time.time()
# Run the model. 5 epochs and 128 batch size (128 samples used for each run through before updating parameters)
trainResults = model.fit(trainX, trainY, epochs=5, batch_size=128, validation_data=(valX, valY), callbacks=[tb])
end = time.time()

print(trainResults.history)


print("Train time: " + str(end - start) + "s")


start = time.time()
# Evaluate on test set
evalResults = model.evaluate(testX, testY, batch_size=128)
end = time.time()

print("test loss, test acc:", evalResults)
print("Inference time: " + str(end - start) + "s")
