
import time
import csv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

t_start = time.time()

print('loading train.npz ...')
d = np.load('data/train.npz')
X, y, classes = d['X'], d['y'], d['classes']
bands = d['bands']
print('  X', X.shape, X.dtype, ' y', y.shape, ' %d classes' % len(classes))

def features(A, chunk=500):
    grid = 4
    n_bands = A.shape[-1]
    out = np.empty((len(A), grid * grid * n_bands * 2), np.float32)

    for i in range(0, len(A), chunk):
        blk = A[i:i + chunk].astype(np.float32)
        n, h, w, b = blk.shape

        regions = blk.reshape(n, grid, h // grid, grid, w // grid, b)

        avg = regions.mean(axis=(2, 4))
        std = regions.std(axis=(2, 4))

        out[i:i + chunk] = np.concatenate(
            [avg.reshape(n, -1), std.reshape(n, -1)], axis=1
        )

    return out

print('extracting features ...')
F = features(X)
print('  F', F.shape)

Ftr, Fva, ytr, yva = train_test_split(
    F, y, test_size=0.2, random_state=0, stratify=y
)

mu, sd = Ftr.mean(axis=0), Ftr.std(axis=0) + 1e-6
Ftr = (Ftr - mu) / sd
Fva = (Fva - mu) / sd

print('training logistic regression ...')
clf = LogisticRegression(max_iter=1000, C=0.1)
clf.fit(Ftr, ytr)

pred = clf.predict(Fva)
acc = accuracy_score(yva, pred)
print('  SPATIAL VALIDATION ACCURACY: %.4f  (%.2f%%)' % (acc, acc * 100))

cm = confusion_matrix(yva, pred)
rec = cm.diagonal() / cm.sum(1)

print('per-class accuracy:')
for i in np.argsort(rec)[::-1]:
    print('  %-22s %6.1f%%' % (classes[i], rec[i] * 100))

print('\nloading test.npz and predicting ...')
t = np.load('data/test.npz')
Xt, ids = t['X'], t['id']
Ft = (features(Xt) - mu) / sd
pt = clf.predict(Ft)

os.makedirs('results', exist_ok=True)

with open('results/submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    for i, p in zip(ids, pt):
        w.writerow([int(i), classes[p]])

print('wrote submission.csv with %d rows' % len(ids))
print('total time: %.0f s' % (time.time() - t_start))
