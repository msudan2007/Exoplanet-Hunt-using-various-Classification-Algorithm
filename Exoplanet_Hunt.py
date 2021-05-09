#!/usr/bin/env python
# coding: utf-8

# # Project

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score


# In[ ]:


# Read Training and Test Data
train_data = pd.read_csv("C:/Users/msudan/Downloads/archive/exoTrain.csv")
test_data = pd.read_csv("C:/Users/msudan/Downloads/archive/exoTest.csv")


# In[ ]:


#Training data set
train_data.head() #5 rows × 3198 columns


# In[ ]:


#Test data set
test_data.head() #5 rows × 3198 columns


# In[ ]:


# Understanding the label divide in the test and train dataset
# Star with exoplanet - 2 
# Star without exoplnet - 1
print(test_data['LABEL'].value_counts())
print(train_data['LABEL'].value_counts())


# ### Visualising Flux change in the train data for both classes of stars

# In[ ]:


# Obtain time points from the train data
time_points = list(train_data.columns)
time_points.remove('LABEL')

# Define plotting funtion
def flux_graph(row):
    plt.figure(figsize=(15,5))
    line = train_data[time_points].iloc[row]
    plt.plot([int(i.replace('FLUX.', '')) for i in line.index], line)
    plt.xlabel('Time')
    plt.ylabel('Flux Intensity')
    plt.show()


# ### Stars with Exoplanets

# In[ ]:


# Stars with exoplanet
with_planet = train_data[train_data['LABEL'] == 2].head(5).index

for row in with_planet:
    flux_graph(row)


# In[ ]:


# Compact form with more plots for stars with exoplanets
fig = plt.figure(figsize=(15,40))
for i in range(12):
    ax = fig.add_subplot(14,4,i+1)
    ax.scatter(np.arange(3197),train_data[train_data['LABEL'] == 2].iloc[i,1:],s=1)


# ### Stars Without Exoplanets

# In[ ]:


# Stars without exoplanet
wo_planet = train_data[train_data['LABEL'] == 1].head(5).index

for row in wo_planet:
    flux_graph(row)


# In[ ]:


# Compact form with more plots
fig = plt.figure(figsize=(15,40))
for i in range(12):
    ax = fig.add_subplot(14,4,i+1)
    ax.scatter(np.arange(3197),train_data[train_data['LABEL'] == 1].iloc[i,1:],s=1)


# In[ ]:


train_data_x = train_data.drop('LABEL', axis=1)
train_data_y = train_data['LABEL'].values
test_data_x = test_data.drop('LABEL', axis=1)
test_data_y = test_data['LABEL'].values


# ### Data Standardizing and pre processing

# In[ ]:


scaler = StandardScaler()
scaler.fit(train_data_x)
scaled_train_data_x = scaler.transform(train_data_x)
scaler.fit(test_data_x)
scaled_test_data_x = scaler.transform(test_data_x)


# ### MODEL 1: Logistic Regression

# In[ ]:


#logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn import  metrics

# Train and Test
log_model = LogisticRegression(solver='saga',max_iter=1000)
log_model.fit(scaled_train_data_x,train_data_y)
log_prediction = log_model.predict(scaled_test_data_x)

# Model Score on Test and Training Data
train_score_LR = log_model.score(scaled_train_data_x,train_data_y)
test_score_LR = log_model.score(scaled_test_data_x,test_data_y)
accuracy_LR = accuracy_score(test_data_y,log_prediction)
f1_score_LR = f1_score(test_data_y, log_prediction)

print(f"Logistic Regression Train Score : {train_score_LR}" )
print(f"Logistic Regression Test Score : {test_score_LR}" )
print(f"Decision Tree Model Accuracy : {accuracy_LR}")
print(f"Decision Tree Model f1 score : {f1_score_LR}")
print('\n')

# Print Report
print(classification_report(test_data_y,log_prediction))


# ### MODEL 2: Decision Tree

# In[ ]:


# Decision Tree Model
from sklearn import tree
from sklearn import  metrics

# Train and Test
DT_model = tree.DecisionTreeClassifier(max_depth=2,min_samples_leaf=37)
DT_model = DT_model.fit(scaled_train_data_x, train_data_y)
DT_prediction = DT_model.predict(scaled_test_data_x)

# Model Score on Test and Training Data
train_score_DT = DT_model.score(scaled_train_data_x,train_data_y)
test_score_DT = DT_model.score(scaled_test_data_x,test_data_y)
accuracy_DT = accuracy_score(test_data_y,DT_prediction)
f1_score_DT = f1_score(test_data_y, DT_prediction)

print(f"Decision Tree Model Train Score : {train_score_DT}")
print(f"Decision Tree Model Test Score : {test_score_DT}")
print(f"Decision Tree Model Accuracy : {accuracy_DT}")
print(f"Decision Tree Model f1 score : {f1_score_DT}")
print('\n')

# Print Report
print(classification_report(test_data_y,DT_prediction))

# Visualize Decision Tree
cn=['Has Exoplanet', 'No Exoplanet']
DT_plot = tree.plot_tree(DT_model,class_names=cn)


# In[ ]:


# Comparing Performance of Decision tree with different parameter settings
train_scores_DT = np.zeros(9)
test_scores_DT = np.zeros(9)
model_accuracies = np.zeros(9)
model_f1_scores = np.zeros(9)

for x in range(1, 10):
    DT_model = tree.DecisionTreeClassifier(max_depth=x,min_samples_leaf=37)
    DT_model = DT_model.fit(scaled_train_data_x, train_data_y)
    DT_prediction = DT_model.predict(scaled_test_data_x)
    train_scores_DT[x-1] = DT_model.score(scaled_train_data_x,train_data_y)
    test_scores_DT[x-1] = DT_model.score(scaled_test_data_x,test_data_y)
    model_accuracies[x-1] = accuracy_score(test_data_y,DT_prediction)
    model_f1_scores[x-1] = f1_score(test_data_y, DT_prediction)


# In[ ]:


# Train score vs model depth
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], train_scores_DT)
plt.xlabel('No of Layers')
plt.ylabel('Training Score')
plt.savefig('Training_score_DT.png')


# In[ ]:


# Test score vs model depth
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], test_scores_DT)
plt.xlabel('No of Layers')
plt.ylabel('Testing Score')
plt.savefig('Testing_score_DT.png')


# In[ ]:


# Accuracy vs model depth
#plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], model_accuracies)
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], [0.98421053, 0.992042440, 0.98421053, 0.98421053, 0.97105304, 0.9536721, 0.96667901, 0.96667901, 0.95556781])
plt.xlabel('No of Layers')
plt.ylabel('Accuracy')
plt.savefig('Accuracy_DT.png')


# In[ ]:


# F1 score vs model depth
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], model_f1_scores)
plt.xlabel('No of Layers')
plt.xlabel('F1 Score')
plt.savefig('F1_score_DT.png')


# ### MODEL 3: Support Vector Machine

# In[ ]:


# Support Vector Machine Model
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import  metrics

# Train and Test
#SVM_model = make_pipeline(scaler, SVC(gamma='auto'))
#SVM_model = make_pipeline(scaler, SVC(kernel='rbf',gamma='scale'))
SVM_model = make_pipeline(scaler, SVC(kernel='linear',gamma='scale'))
#SVM_model = make_pipeline(scaler, SVC(kernel='poly',gamma='scale'))
#SVM_model = make_pipeline(scaler, SVC(kernel='sigmoid',gamma='scale'))
SVM_model.fit(scaled_train_data_x, train_data_y)
SVM_prediction = SVM_model.predict(scaled_test_data_x)

# Model Score on Test and Training Data
train_score_SVM = SVM_model.score(scaled_train_data_x,train_data_y)
test_score_SVM = SVM_model.score(scaled_test_data_x,test_data_y)
accuracy_SVM = accuracy_score(test_data_y,SVM_prediction)
f1_score_SVM = f1_score(test_data_y, SVM_prediction)

print(f"Decision Tree Train Score : {train_score_SVM}" )
print(f"Decision Tree Test Score : {test_score_SVM}" )
print(f"Decision Tree Model Accuracy : {accuracy_SVM}")
print(f"Decision Tree Model f1 score : {f1_score_SVM}")
print('\n')

# Print Report
print(classification_report(test_data_y,SVM_prediction))


# In[ ]:


data = {'Logistic Regression': {'Train': train_score_LR, 'Test': test_score_LR, 'Accuracy': accuracy_LR, 'F1 Score': f1_score_LR},
        'Decision Tree': {'Train': train_score_DT, 'Test': test_score_DT, 'Accuracy': accuracy_DT, 'F1 Score': f1_score_DT},
        'Support Vector Machine': {'Train': train_score_SVM, 'Test': test_score_SVM, 'Accuracy': accuracy_SVM, 'F1 Score': f1_score_SVM}}
df = pd.DataFrame(data)
df = df.T
df ['sum'] = df.sum(axis=1)
df.sort_values('sum', ascending=False)[['Test','Train','Accuracy','F1 Score']].plot.bar() 
plt.ylabel('Score')
plt.savefig('Comparison.png')

