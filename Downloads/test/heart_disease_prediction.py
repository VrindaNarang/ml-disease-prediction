#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler ,LabelEncoder , OneHotEncoder
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.metrics import accuracy_score,classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier





# In[26]:


# load the dataset 

data=pd.read_csv(r"C:\python and machine learning\heart.csv")


# In[28]:


# exploratory data analysis(EDA)

print(data.head())
print(data.info())
print(data.describe())


# In[30]:


# storing all coloumns in a list expect for the heart disease one which we have to predict
list1=data.columns[:-1]
list1  


# In[32]:


# checking for missing values 
print(data.isnull().sum())
# we get no null data values  so we don't need to impute anything 


# In[34]:


# visualizing target variable with respect to age
sns.countplot (x='Sex',data=data)
plt.show()


# In[36]:


# visualizing target variable with respect to age 

# set background style of the plot 
sns.set_style('whitegrid')

#plotting histogram for age 
plt.figure(figsize=(10,6))
sns.histplot(data['Age'] , kde=True , color='red' , bins=10)
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[47]:


# hypothetical interpretation based on histogram :

# The histogram shows a bell-shaped distribution centered around ages 50 to 60.
# There is a peak around the age of 55 , indicating there is a significant number of patients in this age group.
# The distribution is slightly right skewed , suggesting there are more patients older than the mean age.
# The age range of patients varies from about 30 to 77 years.
# There are increasing number of patients in the younger age groups(40-45)
# The older age groups (above 70) , has a depreciating peak which which indicates once someone crosses the age of 70 heart disease is less common.


# In[38]:


# Encoding catagorical figures 
# converting string to numbers 

label_encoders={}
categorical_columns=['Sex','ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for colomn in categorical_columns:
    label_encoders[colomn]=LabelEncoder()
    data[colomn]=label_encoders[colomn].fit_transform(data[colomn])
data




# In[64]:


# visualizing correlations 

fig=plt.figure(figsize=(10,10))
correlation_matrix=data.corr()
sns.heatmap(correlation_matrix , annot=True , cmap='coolwarm')

plt.show()


# In[66]:


# Using Correlation for Feature Reduction 

# Identify strong correlations with Target :
#  - focus on features with high absolute correlation values with target variable.
#  -These features are likely to be more important for prediction.

# Remove redundant feautures 

#  - if two features are highly correlated with each other i.e correlation>0.8 consider removing one of them.
#  -keeping both may not provide additional value and lead to overfitting.


# In[40]:


# Input 

x=data.iloc[:,:-1].values

#Output 
y=data.iloc[:,-1].values 



# In[42]:


x,y


# In[44]:


# Feature selection 
# Selecting the best features using SelectKBest 

from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif,f_classif

selector=SelectKBest(f_classif,k=5) 
X_new=selector.fit_transform(x,y)
print(X_new)


# In[76]:


x


# In[46]:


# feauture scaling ----to remove error that could be there due to presence of units 

scaler=StandardScaler()
scaled_features=scaler.fit_transform(X_new)
scaled_features 


# In[48]:


#splitting the dataset 

X_train,X_test,y_train,y_test=train_test_split(scaled_features,y,test_size=0.2,random_state=0)

#classification algorithms 

classifiers={
    'Logistic Regression':LogisticRegression(),
    'Decision Tree':DecisionTreeClassifier(),
    'Random Forest':RandomForestClassifier(),
    'Support Vector Machine':SVC(),
    'K-Nearest Neighbours':KNeighborsClassifier()
    
}


# In[100]:


# Training and evaluating Classifiers

results={}
for name,clf in classifiers.items():
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test) #prediction is done on test data
    cm = confusion_matrix(y_test,y_pred) #confusion matrix for logistic regression is made result of tested and predicted data .
    print(f'Confusion Matrix for{name}:\n',cm)
    accuracy=accuracy_score(y_test,y_pred)
    results[name]=accuracy
    print(f'{name}Accuracy:{accuracy*100:.2f}%')
    print(classification_report(y_test,y_pred))
    print('_________________________________________________________________________________________')


# In[50]:


# finding the best classifier 

best_classifier=max(results,key=results.get)
print(f'Best Classifier is: {best_classifier} with Accuracy: {results[best_classifier]:.4f}')


# In[52]:


from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt 

# Calculate ROC curve 
fpr,tpr,thresholds=roc_curve(y_test,y_pred)

#Calculate AUC
roc_auc=auc(fpr,tpr)

#plot ROC curve 
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC curve (AUC area={roc_auc}')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlim([0.0,1.0]) 
plt.ylim([0.0,1.0])
plt.xlabel('False positive rate')
plt.ylabel('True Positive Rate')
plt.title('Reciever Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()

# the roc_function calculates the tpr and fpr at various thresholds 
# the auc function calculates the area under roc curve.
# auc ranges from 0.5 to 1 with higher values indicating better classifier perfoermance.


# In[54]:


# optimizing logistic regression 

from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.linear_model import LogisticRegression 






# Define the model
logistic_regression = LogisticRegression()



# define the parameter grid 

param_grid={
    'C':[0.001,0.01,0.1,1,10],
    'penalty':['l1','l2'],
    'solver':['liblinear','saga'],}

# create a logistic regression model 
logistic_regression=LogisticRegression(max_iter=1000)

# create GridSearchCV
grid_search=GridSearchCV(estimator=logistic_regression,param_grid=param_grid,cv=5,scoring='accuracy')

# fit GridSearchCv

grid_search.fit(X_train,y_train)

#get the best parameters and the best score 
best_params=grid_search.best_params_
best_score=grid_search.best_score_

# use the best model to predict 

best_pred=grid_search.predict(X_test)










# In[56]:


print('Best Parameters:',best_params)
print('Best Score:',best_score)


# In[58]:


best_model=grid_search.best_estimator_
print('Best Model:',best_model)


# In[72]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix

# Predict on test set 
y_pred=best_model.predict(X_test)

# calculate performance statistics 
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='weighted')
recall=recall_score(y_test,y_pred,average='weighted')
f1=f1_score(y_test,y_pred,average='weighted')
conf_matrix=confusion_matrix(y_test,y_pred)

print(f'Accuracy:{accuracy:.4f}')
print(f'Precision:{precision:.4f}')
print(f'Recall:{recall:.4f}')
print(f'F1 Score:{f1:.4f}')
print(f'Confusion Matrix:\n',conf_matrix)


# In[76]:


param_grid={
    'max_depth':[None,3,4,5,6,7],
    'min_samples_split':[2,5,7],
    'min_samples_leaf':[1,2,4],
    'criterion':['gini','entropy']
}


# In[82]:


dt=DecisionTreeClassifier(random_state=42)
grid_search=GridSearchCV(dt,param_grid,cv=5,scoring='accuracy',verbose=1)


# In[84]:


grid_search.fit(X_train,y_train)


# In[86]:


best_params=grid_search.best_params_
best_model=grid_search.best_estimator_

print('Best Parameters:',best_params)


# In[88]:


#predict on test set 
y_pred=best_model.predict(X_test)

#calculate performance metrics 
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='weighted')
recall=recall_score(y_test,y_pred,average='weighted')
f1=f1_score(y_test,y_pred,average='weighted')
conf_matrix=confusion_matrix(y_test,y_pred)

print(f'Accuracy:{accuracy:.4f}')
print(f'Precision:{precision:.4f}')
print(f'Recall:{recall:.4f}')
print(f'F1 Score:{f1:.4f}')
print(f'Confusion Matrix:\n',conf_matrix)


# In[94]:


import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree

#Get the best model 
best_model=grid_search.best_estimator_

#Plot the decision tree
plt.figure(figsize=(50,20))
plot_tree(best_model,feature_names=x,filled=True,class_names=['No Disease','Disease'])
plt.title('Decision tree visualization')
plt.show()
          


# In[96]:


# applying Boosting 


# In[98]:


from sklearn.ensemble import AdaBoostClassifier 

#Initialise base decision tree classifier 
base_dt=DecisionTreeClassifier(max_depth=5,random_state=42)

#Initialize Adaboost classifier with decision tree as base estimator 
adaboost_clf=AdaBoostClassifier(base_estimator=base_dt,n_estimators=50,random_state=42)


# In[100]:


adaboost_clf.fit(X_train,y_train)


# In[102]:


#Predict on test set 

y_pred=adaboost_clf.predict(X_test)

#calculate performance 
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='weighted')
recall=recall_score(y_test,y_pred,average='weighted')
f1=f1_score(y_test,y_pred,average='weighted')
conf_matrix=confusion_matrix(y_test,y_pred)

print(f'Accuracy:{accuracy:.4f}')
print(f'Precision:{precision:.4f}')
print(f'Recall:{recall:.4f}')
print(f'F1 Score:{f1:.4f}')
print(f'Confusion Matrix:\n',conf_matrix)


# In[104]:


adaboost_clf.feature_importances_


# In[ ]:




