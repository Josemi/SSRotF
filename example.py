"""
Minimal Installation Verification Example
 
This script provides a simple working example to verify that all dependencies 
are correctly installed and that the Semi-Supervised Rotation Forest library 
is functioning properly.
 
IMPORTANT: This is NOT intended as a performance benchmark or demonstration of 
the method's effectiveness. The dataset used here is deliberately minimal and 
may not represent scenarios where Semi-Supervised Rotation Forest excels.
 
For comprehensive performance comparisons and proper experimental setup, 
please refer to the experiments described in the paper and the full 
experimental scripts in experiments.py.
 
Purpose:
- Verify successful installation of all required packages
- Confirm basic functionality of the library
- Provide a quick sanity check that the code runs without errors
 
Usage:
    python example.py
"""

# Imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sslearn.model_selection import artificial_ssl_dataset
from RotationForestSSL import RotationForestSSLClassifier
from SSLTree import SSLTree

# Comparison method
from sslearn.wrapper import TriTraining
from rotation import RotationForestClassifier

# Metric
from sklearn.metrics import accuracy_score


# Load iris
X,y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Remove labels
X_train_ssl, y_train_ssl, X_unlabeled, y_unlabeled = artificial_ssl_dataset(X_train, y_train, 0.1)

# Train SSRotF
ssrotf = RotationForestSSLClassifier(n_estimators=100, base_estimator=SSLTree(max_depth=100, w=0.85, max_features="sqrt"))
ssrotf.fit(X_train_ssl, y_train_ssl)

# Train comparison method
trirotf = TriTraining(base_estimator=RotationForestClassifier(n_estimators=33,))
trirotf.fit(X_train_ssl, y_train_ssl)

# Predict
y_pred_ssrotf = ssrotf.predict(X_test)
y_pred_trirotf = trirotf.predict(X_test)

# Evaluate
print("SSRotF: " + str(accuracy_score(y_test, y_pred_ssrotf)))
print("Tri(RotF): " + str(accuracy_score(y_test, y_pred_trirotf)))
