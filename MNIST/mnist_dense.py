import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow.keras.layers as tfkl

# Loading MNIST
(trainX, trainY) , (testX, testY) = mnist.load_data()

trainX = np.divide(trainX, 255.0)
testX = np.divide(testX, 255.0)

valX = trainX[-12000:]
valY = trainY[-12000:]
trainX = trainX[:-12000]
trainY = trainY[:-12000]


# Dense NN
# 218058 parameters
dinput = tfkl.Input(shape=(28,28))
# 28 x 28 input img
dflat = tfkl.Flatten()(dinput)
# Flatten (in a FCN inputs need to be 1D like a row of nodes)
ddense2 = tfkl.Dense(256, activation="sigmoid")(dflat)
ddense4 = tfkl.Dense(64, activation="sigmoid")(ddense2)
# A good rule of thumb is to use powers of 2
doutput = tfkl.Dense(10)(ddense4)
# The output is logits directly as loss below is from_logits. Alternatively could set this to softmax

# This model architecture is more complex than it probably needs to be but demonstrates gap between CNN and FCN

denseModel = tf.keras.Model(inputs=dinput, outputs=doutput)

# Print model details
denseModel.summary()


tf.keras.utils.plot_model(
    denseModel,
    to_file="dense_model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="LR",
    expand_nested=False,
    dpi=96,
    show_layer_activations=True
)



# Adam optimizer - standard, optimized vs SGD
# SparseCategoricalCrossEntropy for multi-class classification where labels are integer valued and not encoded
# Accuracy metric
denseModel.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# TensorBoard for visualization
tb = tf.keras.callbacks.TensorBoard(log_dir='dense_logs')


start = time.time()
# Train - 5 epochs and 128 batch size (# of samples in a pass through before updating params)
trainResults = denseModel.fit(trainX, trainY, epochs=5, batch_size=128, validation_data=(valX, valY), callbacks=[tb])
end = time.time()

print(trainResults.history)

print("Train time: " + str(end - start) + "s")

start = time.time()
# Evaluate on test set 
evalResults = denseModel.evaluate(testX, testY, batch_size=128)
end = time.time()

print("test loss, test acc:", evalResults)

print("Inference time: " + str(end - start) + "s")
