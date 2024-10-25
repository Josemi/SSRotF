# Semi-Supervised Rotation Forest
Rotation Forest (RotF) is a popular tree-based ensemble method that has gained a deserved fame in the machine learning community.
Its effectiveness is based on the rotation of the data before the base estimators training. 
This process increases the diversity of the ensemble, which in turn improves its generalization capabilities.
Rotation Forest is a supervised learning (SL) method that should be trained with a sufficient number of labeled instances, in order to obtain a robust model.

The field of machine learning has witnessed a surge of development over the last decades, with the proliferation of novel techniques such as semi-supervised learning (SSL).
SSL lies between supervised and unsupervised learning.
It allows the solution of classification and regression problems characteristic of supervised learning, while also allowing for the acquisition of information from unlabeled instances.
%Presentación SSRotF
This paper presents a semi-supervised version of the ensemble method Rotation Forest for tabular data, called Semi-Supervised Rotation Forest (SSRotF). 
This semi-supervised version includes two main modifications that allow Rotation Forest to use unlabeled data, thereby improving its performance.


In order to demonstrate the performance of the proposed method and to extend the experimentation, tests were conducted using 54 different UCI datasets.
This experimentation also includes a meta-learning analysis in which 16 datasets were selected, for which the good results obtained lead us to believe that they verify some assumptions of the semi-supervised learning, making them more suitable for this and future experimentation.
This experimentation demonstrates the capabilities of Rotation Forest, even in situations with few labeled instances, and highlights the improvement of the proposed semi-supervised version.

## Authors
- José Miguel Ramírez-Sanz
- David Martínez-Acha
- Álvar Arnaiz-González
- César García-Osorio
- Juan J. Rodríguez

### Cite this software as:
Under review