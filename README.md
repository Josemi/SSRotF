# 🌳👀 Semi-Supervised Rotation Forest 🌳👀
 
Rotation Forest (RotF) is a popular tree-based ensemble method that has gained a deserved fame in the machine learning community.  key aspect of its design is the transformation of the feature space through rotation before training
the base learners. This process increases the diversity of the ensemble, which in turn improves its generalization capabilities. Rotation Forest is a supervised learning (SL) method that should be trained with a sufficient number of labeled instances, in order to obtain a robust model.
 
> Rodríguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006). Rotation forest: A new classifier ensemble method. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(10), 1619–1630. [IEEE](https://doi.org/10.1109/TPAMI.2006.211).
 
The field of machine learning has witnessed a surge of development over the last decades, with the proliferation of novel techniques such as semi-supervised learning (SSL). SSL lies between supervised and unsupervised learning. It allows the solution of classification and regression problems characteristic of supervised learning, while also allowing for the acquisition of information from unlabeled instances.  
 
This paper presents a semi-supervised version of the ensemble method Rotation Forest for tabular data, called Semi-Supervised Rotation Forest (SSRotF). This semi-supervised version includes two main modifications that allow Rotation Forest to use unlabeled data, thereby improving its performance.  
 
## 🔬 Experimentation  
 
In order to demonstrate the performance of the proposed method, an extensive experiment was conducted using 54 different UCI datasets. A subset of 16 datasets was identified where the proposed method performs exceptionally, hinting at alignment of these datasets with SSL assumptions.

A meta-learning analysis showed that only four meta-feature metrics are enough to distinguish these cases, suggesting that meta-learning could play a role in predicting dataset suitability for SSL. 

This experimentation demonstrates the RotF capabilities, even in situations with few labeled instances, and highlights the improvement of the proposed semi-supervised version.

### Experiments launch example:
```Bash
python experiments.py
```
 
## 📂 Repository Structure  
 
This repository contains the following files:  
 
### 🔹 RotationForestSSL.py  
This file implements *Semi-Supervised Rotation Forest (SSRotF)*, introducing two modifications to the original Rotation Forest algorithm:
1. *Modified PCA transformation* → PCA is applied by incorporating a subset of the unlabeled instances, thereby increasing the model’s diversity.  
2. *SSLTree as base estimator* → Instead of using traditional decision trees, SSRotF employs *SSLTree*, a decision tree that is intrinsically semi-supervised.  
 
### 🔹 SSLTree.py  
This file contains the implementation of *SSLTree*, a CART decision tree model specifically designed for semi-supervised learning.  
Unlike standard decision trees, *SSLTree incorporates unlabeled instances during its construction*, allowing it to extract additional information from the data distribution.
 
This method is based on the impurity calculation described in:
> Levatić, J., Ceci, M., Kocev, D., & Džeroski, S. (2017). Semi-supervised classification trees. Journal of Intelligent Information Systems, 49, 461–486. [Springer](https://doi.org/10.1007/s10844-017-0457-4).  
 
### 🔹 experiments.py  
This script provides the *experimental framework* for evaluating SSRotF and comparing it with other models:
 
- *Cross-validation with repetition* → Each experiment runs multiple repetitions, and within each repetition, cross-validation is applied. This ensures robust evaluation by reducing the impact of randomness. 
- The experiments are executed by repetition, within each repetition by dataset, within each dataset by fold, and within each fold by label proportion. Each of these configurations is stored and then executed using a pool of jobs in multiprocessing.
- The utilities used throughout the script (*utils* package) must be requested, as they include code used in multiple other experiments and are considered for private use. The utils package is primarily used for data collection and handling datasets (p.e, metrics calculation).

### 🔹 utils.py
Utils file with the implementation of functions for loading data and saving results. Note: The path to the UCI data must be set in this file.

### 🔹 requirements.txt
These are the requirements needed to run the experiments. They are not the requirements for the proposed method. To run the original Rotation Forest, you need the implementation from [Admirable-Methods](https://github.com/jlgarridol/admirable-methods/blob/master/ubulearn/rotation.py) repository.

### 🔹 results.pdf  
A PDF file with the results of the above experiments, for each label proportion and metric.

### 🔹 results.csv  
A CSV file with the results of the above experiments, for each label proportion and metric.

## ⚙️ Execution Example
```Python
# Imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sslearn.model_selection import artificial_ssl_dataset
from RotationForestSSL import RotationForestSSLClassifier
from SSLTree import SSLTree

# Load iris
X,y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Remove labels
X_train_ssl, y_train_ssl, X_unlabeled, y_unlabeled = artificial_ssl_dataset(X_train, y_train, 0.1)

# Train SSRotF
ssrotf = RotationForestSSLClassifier(n_estimators=100, base_estimator=SSLTree(max_depth=100, w=0.85, max_features="sqrt"))
ssrotf.fit(X_train_ssl, y_train_ssl)

# Predict
y_pred = ssrotf.predict(X_test)
```
 
## 👥 Authors  
 
- José Miguel Ramírez-Sanz  
- David Martínez-Acha  
- Álvar Arnaiz-González  
- César García-Osorio  
- Juan J. Rodríguez  
 
### 📌 Cite this software as:  
Under review
