import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Image size and batch
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Data Augmentation
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Load EfficientNetB0
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights="imagenet"
)

base_model.trainable = False  # Freeze layers

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_aug(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/ham10000_efficientnet.h5")

print("\n🎉 Training completed successfully!")
print("📁 Your model is saved at: model/ham10000_efficientnet.h5")
