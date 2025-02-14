''' Isolet Dataset '''

''' Description (source: https://doi.org/10.24432/C51G69): '''
# This data set was generated as follows. 150 subjects spoke the name of each 
# letter of the alphabet twice. Hence, we have 52 training examples from each 
# speaker. The speakers are grouped into sets of 30 speakers each, 
# and are referred to as isolet1, isolet2, isolet3, isolet4, and isolet5. 
# The data appears in isolet1+2+3+4.data in sequential order, 
# first the speakers from isolet1, then isolet2, and so on.  
# The test set, isolet5, is a separate file.
# You will note that 3 examples are missing.  
# I believe they were dropped due to difficulties in recording.
    

''' Source: '''
# Ron Cole and Mark Fanty. ISOLET. UCI Machine Learning Repository, 1994.
# DOI: https://doi.org/10.24432/C51G69

''' ---------------------------------------------------------------------- '''

# Instructions on downloading "ucimlrepo"
# can be found in the link under Source.

from ucimlrepo import fetch_ucirepo
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC
import tensorflow as tf
import time
# import pandas as pd

# Import Isolet Dataset
isolet_data = fetch_ucirepo(id = 54)

# Separating features and classes
# "features" and "classes" are already pandas dataframes
features = isolet_data.data.features
classes = isolet_data.data.targets

''' --------------------------------------------------------------------- '''

# Descriptions of Features
''' (CALCULATING CORRELATION MATRIX TAKES TOO MUCH COMPUTAITONAL EFFORT) '''
# Correlation matrix visualization 
# corr_matrix = features.corr()
# plt.figure()
# sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')
# plt.title("Correlation Matrix of ISOLET features")
# plt.show()

# Somewhat difficult difficult to interpret as there are many features
feat_description = features.describe()
feat_description.head(5)
# It does appear that the data is numerical 
# and normalized between -1 and 1

# Let's check for NaN's or missing entries
features.isna().values.any()
# There are currently NaN's or missing entries as described above.

# Check for inf's
inf_count = np.isinf(features).values.sum()
# No inf's either - data is clean and normalized.

''' --------------------------------------------------------------------- '''

# Descriptions of Classes
class_count = classes.nunique()
class_description = classes.value_counts() 
# Each class is associated with 300 samples except 
# class 6 and 13, which have 298 and 299 samples, respectively.  
# Although we could stratify our training/testing data based on these
# count differences, it may not be entirely necessary.  We'll do 
# a classic 70/30 split across the board at random. 

''' --------------------------------------------------------------------- '''

# Training Section

# Training/Testing data splits
x_train,x_test,y_train,y_test = train_test_split(features,classes,
                                                 test_size = 0.3,
                                                 random_state = 2356)
# Converting to 1D arrays for training
# We subtracted 1 so the classes will start from 0 instead of 1
y_train = y_train.values.ravel()-1
y_test = y_test.values.ravel()-1


# kNN Classifier
knn_time = time.time()
knn_classifier = kNN(10)
knn_classifier.fit(x_train,y_train)
knn_time = time.time() - knn_time

# SVM - Linear Kernel - One vs One
svml_time = time.time()
svm_classifier_linear = SVC(kernel = 'linear', probability = True, random_state = 2356)
svm_classifier_linear.fit(x_train,y_train)
svml_time = time.time() - svml_time

# SVM - Radial Basis Fucntion Kernel - One vs One
svmrbf_time = time.time()
svm_classifier_rbf = SVC(kernel = 'rbf', probability = True, random_state = 2356)
svm_classifier_rbf.fit(x_train,y_train)
svmrbf_time = time.time() - svmrbf_time

# Random Forest
rf_time = time.time()
rf_classifier = rf(100)
rf_classifier.fit(x_train,y_train)
rf_time = time.time() - rf_time

# Fully Connected Neural Network
nn_classifier = tf.keras.models.Sequential([
    tf.keras.layers.Dense(len(x_train), activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256), 
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(class_count)
    ])

objective_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
nn_classifier.compile(optimizer = 'adam',
                      loss = objective_function,
                      metrics = ['accuracy'])

nn_time = time.time()
nn_classifier.fit(x_train,y_train,batch_size = 32,epochs = 20,verbose = 1)
nn_time = time.time() - nn_time

''' -------------------------------------------------------------------- '''

# Testing Section
knn_pred = knn_classifier.predict(x_test)
svml_pred = svm_classifier_linear.predict(x_test)
svmrbf_pred = svm_classifier_rbf.predict(x_test)
rf_pred = rf_classifier.predict(x_test)

# Checking accuracy
knn_acc = accuracy_score(y_test,knn_pred)
svml_acc = accuracy_score(y_test,svml_pred)
svmrbf_acc = accuracy_score(y_test,svmrbf_pred)
rf_acc = accuracy_score(y_test,rf_pred)
nn_acc = nn_classifier.evaluate(x_test,y_test)[1]

print("Testing accuracies of the following models")
print(f"3NN Accuracy: {knn_acc}")
print(f"SVML Accuracy: {svml_acc}")
print(f"SVMRBF Accuracy: {svmrbf_acc}")
print(f"Random Forest Accuracy: {rf_acc}")
print(f"Neural Network Accuracy: {nn_acc}")

print("Training times for each classifier in secs")
print(f"3NN Train Time:  {knn_time}")
print(f"SVML Train Time:  {svml_time}")
print(f"SVMRBF Train Time:  {svmrbf_time}")
print(f"Random Forest Train Time:  {rf_time}")
print(f"Neural Network Train Time:  {nn_time}")      
      




