import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time
import csv

EPOCHS = 15
 
t_start = time.time()

# 1. Load

# Load the data
data = np.load('../data/train.npz')
satellite_images, land_cover_labels = data['X'].astype('float32'), data['y']
classes = data['classes']

# Normalize pixel values to be between 0 and 1
satellite_images, land_cover_labels = satellite_images / 255.0, land_cover_labels

# Split the data into train and test datasets
train_images, test_images, train_labels, test_labels = train_test_split(satellite_images, land_cover_labels, test_size=0.2)

train_images = satellite_images
train_labels = land_cover_labels


del data

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.25, fill_mode="reflect"),
], name="augment")

# 4. Train

model = keras.models.Sequential([
    keras.Input(shape=train_images.shape[1:]),    # (64, 64, 13)
    augment,
    keras.layers.Conv2D(32, (3, 3), activation='relu'), # 13 spectral bands
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10) # 10 different classes in the EuroSAT dataset
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

model.save('cnn.keras')

# Predict with test data
print('\nloading test.npz and predicting ...')
t = np.load('../data/test.npz')
Xt, ids = t['X'], t['id']
Xt = Xt / 255.0          # same scaling as the training images

pt = model.predict(Xt, batch_size=256, verbose=0).argmax(axis=1)
# no manual (Xt - mu) / sd here: the model's Normalization layer does it


with open('../results/submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    for i, p in zip(ids, pt):
        w.writerow([int(i), classes[p]])

print('wrote submission.csv with %d rows' % len(ids))
print('total time: %.0f s' % (time.time() - t_start))





plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()


from eval_metrics import report_f1, plot_confusion_percent

y_pred = model.predict(X_val, batch_size=256, verbose=0).argmax(axis=1)
report_f1(y_val, y_pred, classes, csv_path='../results/f1_report.csv')
plot_confusion_percent(y_val, y_pred, classes, save_path='../results/confusion_matrix.png')

