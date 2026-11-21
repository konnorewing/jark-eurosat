import csv
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from eval_metrics import report_f1, plot_confusion_percent

# 1. Load the saved model
model = keras.models.load_model('cnn.keras')
model.summary()

# 2. Scale inputs the way training did:
#    a model with a Normalization layer takes raw values; the original model expects /255
has_norm = any(isinstance(l, keras.layers.Normalization) for l in model.layers)

def prep(X):
    X = X.astype('float32')
    return X if has_norm else X / 255.0

# 3. Rebuild the same validation split (test_size, stratify and random_state must match training)
data = np.load('../data/train.npz')
X, y, classes = data['X'], data['y'], data['classes']
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
del X, data

y_pred = model.predict(prep(X_val), batch_size=256, verbose=0).argmax(axis=1)
report_f1(y_val, y_pred, classes)
plot_confusion_percent(y_val, y_pred, classes, gamma=0.4)
plt.show()

# 4. Predict test.npz and write the submission
t = np.load('../data/test.npz')
pt = model.predict(prep(t['X']), batch_size=256, verbose=0).argmax(axis=1)

with open('../results/submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    w.writerows([int(i), classes[p]] for i, p in zip(t['id'], pt))
print('wrote submission.csv with %d rows' % len(pt))