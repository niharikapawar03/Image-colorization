# Install required libraries
!pip install tensorflow tensorflow_datasets matplotlib

# Import necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Resizing
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load and preprocess a small subset of the dataset
def load_small_dataset(num_samples=200):
    dataset, info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
    train_data, test_data = dataset['train'], dataset['test']

    def preprocess(image, label):
        image = tf.image.resize(image, (128, 128)) / 255.0  # Normalize to [0, 1]
        gray_image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
        return gray_image, image  # Input: Grayscale, Target: RGB

    # Limit the dataset to `num_samples` images for quick testing
    train_data = train_data.take(num_samples).map(preprocess).batch(16).prefetch(1)
    test_data = test_data.take(num_samples // 4).map(preprocess).batch(16).prefetch(1)

    return train_data, test_data

train_data, test_data = load_small_dataset(num_samples=100)

# Build the model
def build_model():
    input_layer = Input(shape=(128, 128, 1))  # Grayscale input
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)  # Upsample to 256x256
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)  # Upsample to 512x512
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)  # Output RGB
    output_layer = Resizing(128, 128)(x)  # Resize back to 128x128
    return Model(input_layer, output_layer)

model = build_model()
model.compile(optimizer="adam", loss="mse")

# Train the model on a small subset of the dataset
history = model.fit(train_data, epochs=5, validation_data=test_data)  # Fewer epochs for quick testing

# Evaluate the model on the small test set
for gray_image, true_color in test_data.take(1):
    predicted_color = model.predict(gray_image)

    # Visualize results
    def show_images(gray, true_color, predicted_color, num_images=5):
        for i in range(min(num_images, len(gray))):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Grayscale")
            plt.imshow(tf.squeeze(gray[i]), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("True Color")
            plt.imshow(true_color[i])
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Predicted Color")
            plt.imshow(predicted_color[i])
            plt.axis("off")

            plt.show()

    show_images(gray_image, true_color, predicted_color)
