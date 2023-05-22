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


# CNN architecture
# 666 parameters
input = tfkl.Input(shape=(28,28,1))
# 28 x 28 image input
c1 = tfkl.Conv2D(4, 3, activation="leaky_relu")(input)
# Correlation 3x3 filter, expanding to 2 channels. Activation is leaky_relu (good for imgs)
p1 = tfkl.MaxPool2D(3)(c1)
# Max pooling - take the maximum value from a 3x3 window
c2 = tfkl.Conv2D(8, 3, activation="leaky_relu")(p1)
p2 = tfkl.MaxPool2D(3)(c2)
flat = tfkl.Flatten()(p2)
output = tfkl.Dense(10)(flat)
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
tb = tf.keras.callbacks.TensorBoard(log_dir='cnn_logs')


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


# Get feature maps from one input image

plt.imshow(testX[0], cmap='gray')
plt.savefig('orig_mnist_img.png')
plt.show()


firstConvModel = tf.keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
firstMaps = firstConvModel.predict(testX[:1])
for i in range(4):
    plt.imshow(firstMaps[0,:,:,i], cmap='gray')
    plt.savefig('conv1filter' + str(i) + '.png')
    plt.show()
firstPoolModel = tf.keras.Model(inputs=model.inputs, outputs=model.layers[2].output)
firstPoolMaps = firstPoolModel.predict(testX[:1])
for i in range(4):
    plt.imshow(firstPoolMaps[0,:,:,i], cmap='gray')
    plt.savefig('pool1filter' + str(i) + '.png')
    plt.show()
secConvModel = tf.keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
secMaps = secConvModel.predict(testX[:1])
for i in range(8):
    plt.imshow(secMaps[0,:,:,i], cmap='gray')
    plt.savefig('conv2filter' + str(i) + '.png')
    plt.show()
secPoolModel = tf.keras.Model(inputs=model.inputs, outputs=model.layers[4].output)
secPoolMaps = secPoolModel.predict(testX[:1])
for i in range(8):
    plt.imshow(secPoolMaps[0,:,:,i], cmap='gray')
    plt.savefig('pool2filter' + str(i) + '.png')
    plt.show()



