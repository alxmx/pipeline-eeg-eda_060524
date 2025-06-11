import matplotlib
print(f"Matplotlib backend: {matplotlib.get_backend()}")
print(f"Matplotlib version: {matplotlib.__version__}")

import matplotlib.pyplot as plt
import os

# Test simple PDF save
os.makedirs('reports', exist_ok=True)

# Test 1: Simple plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_title('Test Plot')

test_path = os.path.join('reports', 'test_simple_plot.pdf')
try:
    fig.savefig(test_path)
    print(f"Simple plot saved successfully: {os.path.exists(test_path)}")
except Exception as e:
    print(f"Error saving simple plot: {e}")
finally:
    plt.close(fig)

# Test 2: Check sklearn availability
try:
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    print("sklearn.metrics imports successful")
except ImportError as e:
    print(f"sklearn.metrics import error: {e}")

# Test 3: Test with sample data
try:
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Test confusion matrix creation
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(svm, X_test, y_test, ax=ax_cm, cmap='Blues')
    plt.title('Test Confusion Matrix')
    
    cm_path = os.path.join('reports', 'debug_confusion_matrix.pdf')
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)
    
    print(f"Confusion matrix file exists: {os.path.exists(cm_path)}")
    if os.path.exists(cm_path):
        print(f"File size: {os.path.getsize(cm_path)} bytes")
    
except Exception as e:
    print(f"Error in sklearn test: {e}")
    import traceback
    traceback.print_exc()
