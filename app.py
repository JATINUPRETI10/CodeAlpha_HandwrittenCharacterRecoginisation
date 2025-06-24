import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import gradio as gr
import cv2

# Load and preprocess MNIST
(xm_train, ym_train), (xm_test, ym_test) = tf.keras.datasets.mnist.load_data()
xm_train = xm_train.astype('float32') / 255.0
xm_test = xm_test.astype('float32') / 255.0
xm_train = np.expand_dims(xm_train, -1)
xm_test = np.expand_dims(xm_test, -1)
ym_train = ym_train.astype('int64')
ym_test = ym_test.astype('int64')

mnist_train_ds = tf.data.Dataset.from_tensor_slices((xm_train, ym_train)).batch(128).prefetch(1)
mnist_test_ds = tf.data.Dataset.from_tensor_slices((xm_test, ym_test)).batch(128).prefetch(1)

# Load EMNIST Letters and preprocess
(ds_e_train, ds_e_test), _ = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True, with_info=True)

def preprocess_emnist(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image)
    image = tf.reshape(image, [28, 28, 1])
    label = tf.cast(label + 9, tf.int64)
    return image, label

ds_e_train = ds_e_train.map(preprocess_emnist).batch(128).prefetch(1)
ds_e_test = ds_e_test.map(preprocess_emnist).batch(128).prefetch(1)

# Merge datasets
train_ds = mnist_train_ds.concatenate(ds_e_train)
test_ds = mnist_test_ds.concatenate(ds_e_test)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(36, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=3, validation_data=test_ds)

# Class labels (0–9 and A–Z)
labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

# Prediction function
def predict(img):
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # invert
    img = img / 255.0
    img = tf.image.transpose(img)
    img = img.numpy().reshape(1, 28, 28, 1)
    pred = model.predict(img, verbose=0)[0]
    return {labels[i]: float(pred[i]) for i in range(36)}

# Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(canvas_size=(280, 280)),
    outputs=gr.Label(num_top_classes=5),
    title="Handwritten Character Recognizer (0–9 + A–Z)"
).launch()
