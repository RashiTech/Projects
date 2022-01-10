#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Classifying breast cancer as malign/ benign
import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
#loading data
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0], breast_cancer_data.feature_names)
print(breast_cancer_data.target, breast_cancer_data.target_names)

#splitting training data into train and test data
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state =100 )
print(len(training_data), len(training_labels))

# creating classifier
k_list = []
accuracies = []
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  print(k, classifier.score(validation_data, validation_labels))
  k_list.append(k)
  accuracies.append(classifier.score(validation_data, validation_labels))
plt.plot(k_list,accuracies)
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.show()

