import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Test if the individual PDF generation works
os.makedirs('reports', exist_ok=True)
X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple SVM
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

# Test confusion matrix
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(svm, X_test, y_test, ax=ax_cm, cmap='Blues')
plt.title('Test Confusion Matrix')
cm_path = os.path.join('reports', 'test_confusion_matrix.pdf')
fig_cm.savefig(cm_path)
plt.close(fig_cm)

# Test ROC curve
fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(svm, X_test, y_test, ax=ax_roc)
plt.title('Test ROC Curve')
roc_path = os.path.join('reports', 'test_roc_curve.pdf')
fig_roc.savefig(roc_path)
plt.close(fig_roc)

print(f'Confusion matrix saved: {os.path.exists(cm_path)}')
print(f'ROC curve saved: {os.path.exists(roc_path)}')
print(f'Confusion matrix path: {cm_path}')
print(f'ROC curve path: {roc_path}')
