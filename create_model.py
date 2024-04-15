import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dữ liệu từ file numpy đã lưu trước đó
images = np.load("images_gray_thresh_30_40.npy")
labels = np.load("labels_gray_thresh_30_40.npy")


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu ảnh về khoảng [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode nhãn
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 30, 1), padding="SAME"),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.15),

        layers.Conv2D(64,(3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.2),

        layers.Conv2D(128,(3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.2),

        layers.Flatten(),

        layers.Dense(1000,activation='relu'),
        layers.Dense(256,activation='relu'),
        layers.Dense(35,activation='softmax'),
])

# Compile mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(train_images, train_labels, epochs=5, batch_size=16, validation_data=(test_images, test_labels))

model.save('model_gray_thresh_30_40.h5')