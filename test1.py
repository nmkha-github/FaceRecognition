import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import keras.utils

# Step 1: Prepare the training data
# Define the path to the face data directory
data_dir = 'face_data'

# Load the face images and labels
X_train = []
y_train = []
for person_name in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person_name)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        X_train.append(tf.keras.utils.img_to_array(image))
        y_train.append(person_name)

# Convert the face images and labels to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Show image
# plt.imshow(X_train[0].astype('uint8'), interpolation='nearest')
# plt.show()

# Label-encode the person names
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Preprocess the training data for the VGG-Face network
X_train = np.array([utils.preprocess_input(x) for x in X_train])

# Show image
# plt.imshow(X_train[0].astype('uint8'), interpolation='nearest')
# plt.show()

# Step 2
# Define the VGG-Face model as the feature extraction backbone
backbone = VGGFace(model='vgg16', include_top=False,
                   input_shape=(224, 224, 3), pooling='avg')

# Define the full face recognition model
model = Sequential([
    backbone,
    Dropout(0.5),
    Dense(len(os.listdir(data_dir)), activation='softmax')
])

# Step 4: Compile the model
# Compile the face recognition model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Step 5: Train the model
# Convert the training labels to one-hot encoded vectors
y_train_onehot = keras.utils.np_utils.to_categorical(
    y_train, num_classes=len(os.listdir(data_dir)))

# Train the face recognition model
model.fit(X_train, y_train_onehot, batch_size=32,
          epochs=10, validation_split=0.2)

# Step 6: Predict
# Load image
image_path = 'D:/PPNCKH/test/1.jpg'
image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
x = tf.keras.utils.img_to_array(image)
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x)

# Use the trained model to make a prediction
prediction = model.predict(x)

print(prediction)
