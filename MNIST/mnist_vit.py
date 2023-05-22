import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow.keras.layers as tfkl

# Implementing Vision Transformers in Tensorflow
# Followed this tutorial: https://keras.io/examples/vision/image_classification_with_vision_transformer/


class Patches(tfkl.Layer):
    # Class for splitting image into patches of a given size
    def __init__(self, size):
        super().__init__()
        # Size for images to be split into
        self.size = size

    def call(self, imgs):
        # Main method to be called when running the layer
        batch = tf.shape(imgs)[0] # Get the batch size of the images
        # Use built into patch extraction method
        patches = tf.image.extract_patches(
            images=imgs, # imgs to extract
            sizes=[1, self.size, self.size, 1], # sizes of patches - 1 for the batch, size for x,y, 1 for the channel/depth
            strides=[1, self.size, self.size, 1], # size of strides - 1 for batch, size for x,y (don't want overlapping patches), 1 for the channel/depth
            rates = [1,1,1,1], # Get the patches at a rate of 1 for all dimensions (dont skip any)
            padding="VALID", # Only include patches if it is fully contained by the image - no partial patches
        )
        patchdim = patches.shape[-1] # Get the last dimension of the patches obj
        patches = tf.reshape(patches, [batch, -1, patchdim]) # Reshape the patches to 3d tensor instead of 4
        return patches

class PatchEncoder(tfkl.Layer):
    # Class for positional encoding of patches (order of the patches matters)
    def __init__(self, patchnum, proj_dim):
        super().__init__()
        # Number of patches
        self.patchnum = patchnum
        # Projection to a dense layer with proj_dim units
        self.proj = tfkl.Dense(proj_dim)
        # Embedding layer - project from number of patches to projection units
        self.position_embedding = tfkl.Embedding(
            input_dim=patchnum, output_dim=proj_dim
        )

    def call(self, patch):
        # Create a list of positions 0 - patchnum
        positions = tf.range(start=0, limit=self.patchnum, delta=1)
        # Encode the patch by combining the result of putting the patch through the dense layer and
        # the position embedding
        encoded = self.proj(patch) + self.position_embedding(positions)
        return encoded


def vit_classifier():
    projdim = 64 # Arbitrary projdim based off tutorial
    input = tfkl.Input(shape=(28,28,1))
    patches = Patches(7)(input)
    encoded = PatchEncoder(16, projdim)(patches)
    # Create 7x7 patches and encode and project into projdim units
    # 16 7x7 patches in a 28x28 img

    # Transformer block - only using 1
    # Layer normalization of the encoded projection
    norm = tfkl.LayerNormalization(epsilon=1e-6)(encoded)
    # Multihead self-attention of the result
    attention = tfkl.MultiHeadAttention(
        num_heads=4, key_dim=projdim , dropout=0.1
    )(norm, norm)
    # Skip connection: Combine result of attention and encoded projection
    skip1 = tfkl.Add()([attention, encoded])
    # Feed into another layer normalization
    norm2 = tfkl.LayerNormalization(epsilon=1e-6)(skip1)
    # Into dense network: with 2 * proj_dim units
    mlpout = tfkl.Dense(projdim*2, activation=tf.nn.gelu)(norm2)
    # 50% percent dropout of the connections from the dense layer
    mlpoutd = tfkl.Dropout(0.5)(mlpout)
    # Second dense layer with half the units of the prior
    mlpout2 = tfkl.Dense(projdim, activation=tf.nn.gelu)(mlpoutd)
    # 50 percent dropout again
    mlpoutd2 = tfkl.Dropout(0.5)(mlpout2)
    # Skip connection combnining output of dense section and the result of the previous skip
    encoded = tfkl.Add()([mlpoutd2, skip1])
    # Named encoded as this is one transformer block, so allows it to be looped and repeated

    # Final layer normalization
    rep = tfkl.LayerNormalization(epsilon=1e-6)(encoded)
    # Flatten to feed into dense network
    rep2 = tfkl.Flatten()(rep)
    # 50% dropout
    rep3 = tfkl.Dropout(0.5)(rep2)

    # Dense downsampling (2048 -> 1024 -> 10 with 50% dropout)
    features = tfkl.Dense(2048)(rep3)
    featuresd = tfkl.Dropout(0.5)(features)
    features2 = tfkl.Dense(1024)(featuresd)
    features2d = tfkl.Dropout(0.5)(features2)

    output = tfkl.Dense(10)(features2d)

    return tf.keras.Model(inputs=input, outputs=output)





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


# ViT architecture
model = vit_classifier()


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
tb = tf.keras.callbacks.TensorBoard(log_dir='vit_logs')


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

