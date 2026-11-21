"""
ASEN 6337 - EuroSAT competition - BASELINE
==========================================

A deliberately simple model. It does NOT use a neural network and it does NOT
use the spatial arrangement of the pixels at all. It asks one question of each
of the 13 bands: "how bright is this patch on average?"

That is 13 numbers per patch, fed to logistic regression.

It is weak on purpose. It knows the average colour of a patch and nothing else:
not how varied the patch is, not how anything is arranged. Beating it is meant
to be possible. Understanding WHY it fails where it does is the useful part.

The point is to give you a score to beat and a worked example of the whole path:
load, features, train, validate, predict, submission.csv

Runs in well under a minute on a laptop CPU. No GPU needed.
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

t_start = time.time()

# ----------------------------------------------------------------- 1. load
print('loading train.npz ...')
d = np.load('data/train.npz')
X, y, classes = d['X'], d['y'], d['classes']
bands = d['bands']
print('  X', X.shape, X.dtype, ' y', y.shape, ' %d classes' % len(classes))


# ------------------------------------------------------------- 2. features
def features(A, chunk=2000):
    """Per-band mean -> 13 numbers per patch.

    Done in chunks so we never hold a float64 copy of the whole cube in RAM.
    """
    out = np.empty((len(A), A.shape[-1]), np.float32)
    for i in range(0, len(A), chunk):
        blk = A[i:i + chunk].astype(np.float32)
        out[i:i + chunk] = blk.mean(axis=(1, 2))
    return out


print('extracting features ...')
F = features(X)
print('  F', F.shape)

# ------------------------------------------------- 3. hold out a validation set
# The real test set has no labels, so we carve a validation set out of train
# to get an honest estimate before submitting.
Ftr, Fva, ytr, yva = train_test_split(F, y, test_size=0.2,
                                      random_state=0, stratify=y)

# ----------------------------------------------------------------- 4. train
# Logistic regression needs its inputs on a comparable scale, otherwise the
# bands with the largest raw numbers dominate. Note where mu and sd come from:
# the TRAINING half only. Computing them over everything, including the data
# you are about to be judged on, is a leak, and it is a marked criterion.
mu, sd = Ftr.mean(axis=0), Ftr.std(axis=0) + 1e-6
Ftr = (Ftr - mu) / sd
Fva = (Fva - mu) / sd

print('training logistic regression ...')
clf = LogisticRegression(max_iter=400)
clf.fit(Ftr, ytr)

# -------------------------------------------------------------- 5. validate
pred = clf.predict(Fva)
acc = accuracy_score(yva, pred)
print()
print('=' * 46)
print('  BASELINE VALIDATION ACCURACY: %.4f  (%.2f%%)' % (acc, acc * 100))
print('=' * 46)
print()

accuracy_dict = {"overall": acc * 100}

cm = confusion_matrix(yva, pred)
rec = cm.diagonal() / cm.sum(1)
print('per-class accuracy:')
for i in np.argsort(rec)[::-1]:
    print('  %-22s %6.1f%%' % (classes[i], rec[i] * 100))
    accuracy_dict[classes[i]] = rec[i] * 100

pd.DataFrame([accuracy_dict]).to_csv('results/baseline_accuracy.csv', index=False)

# ---------------------------------------------- 6. predict the real test set
print('\nloading test.npz and predicting ...')
t = np.load('data/test.npz')
Xt, ids = t['X'], t['id']
Ft = (features(Xt) - mu) / sd          # the SAME mu and sd as training
pt = clf.predict(Ft)

# ------------------------------------------------------- 7. write submission
import csv
with open('results/submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    for i, p in zip(ids, pt):
        w.writerow([int(i), classes[p]])

print('wrote submission.csv with %d rows' % len(ids))
print('total time: %.0f s' % (time.time() - t_start))
print("""
WHERE TO GO FROM HERE
---------------------
This baseline throws away everything except average brightness. Ideas:

  * Add back how much each band VARIES across the patch, not just its mean.
    That one change is worth several points on its own. Ask yourself why.
  * Use the spatial structure. Texture separates Highway from Residential;
    this model cannot see either.
  * Choose your bands. B10 (cirrus) is nearly empty over land - check it.
    B11/B12 (SWIR) are strong. Justify whatever you keep.
  * Build indices. NDVI = (B08 - B04) / (B08 + B04) is one number that
    already ranks vegetation against water and concrete.
  * Try a CNN, but only after you can explain why this baseline fails where
    it does. The confusion matrix above tells you where to look first.
""")
