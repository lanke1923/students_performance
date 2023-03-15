# %% [markdown]
# LOGISTIC REGRESSION ON STUDENTS' PERFOMANCE DATASET

# %% [markdown]
# Importing the necessary packages for this project.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings("ignore")
import scipy 

# %% [markdown]
# Loading the dataset

# %%
data = pd.read_csv("StudentsPerformance.csv")
data.head(20)

# %% [markdown]
# Exploratory Data Analysis

# %%
data.info()

# %%
data.describe()

# %% [markdown]
# We create a new column for the mean score of the three subjects to ease in identifying the overall performance.

# %%
data['mean score']= ((data['reading score']+ data['writing score'] + data['math score'])/3).round()
data.tail()

# %%
data['gender'].value_counts()

# %% [markdown]
# Label encoding: We convert the categorical data into numeric data for ease in performing the regression analysis.

# %%
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['lunch'] = le.fit_transform(data['lunch'])
data['parental level of education'] = le.fit_transform(data['parental level of education'])
data['race/ethnicity'] = le.fit_transform(data['race/ethnicity'])
data['test preparation course'] = le.fit_transform(data['test preparation course'])
data.head(20)

# %% [markdown]
# Analysing gender and race/ethnicity

# %%
sns.countplot(x= data['gender'], hue=data['race/ethnicity'])
#1=male, 0=female

# %% [markdown]
# 2 = group c,  there are more females from group c. Group c has the highest population.

# %% [markdown]
# Analysing test preparation course

# %%
data['test preparation course'].value_counts()
#0 = completed
#1 =none

# %% [markdown]
# Let us plot a piechart to visulaize this finding.

# %%
plt.pie(data['test preparation course'].value_counts(), 
    labels= ['None', 'Completed'], colors = ['green', 'red'])

# %%
sns.barplot(x='test preparation course', y= 'mean score', data=data)
plt.show()

# %% [markdown]
# From the barplot above it is clear that the students who completed the test preparation course had a higher mean score compared to those who did not.

# %% [markdown]
# Analysing lunch variable

# %%
sns.barplot(x='lunch', y='mean score', data = data)
plt.show()
# 1= standard, 0= free/reduced

# %% [markdown]
# Students who got a standard lunch scored higher.

# %% [markdown]
# Analysing parental level of education variable

# %%
sns.barplot(x=data['parental level of education'], y=data['mean score'])
plt.show()
#0= Associate's degree, 1= Bachelor's degree, 2= highschool, 3= master's degree, 4= some college, 5= some highschool

# %% [markdown]
# The highest performing students had parents with above college level education.

# %% [markdown]
# Now after analysing the categorical columns, we can analyse the relationships between the columns. Let us use a pairplot and a heatmap for correlation.

# %%
plt.figure(figsize=(12,6))
sns.pairplot(data)
plt.show()

# %%
plt.figure(figsize=(12,6))
sns.heatmap(data.corr())
plt.show()

# %% [markdown]
# Data Preprocessing

# %%
data=data.drop(['math score','reading score','writing score'],axis=1)
data.head()

# %%
from sklearn.model_selection import train_test_split
y=data['mean score']
x= data.drop(['mean score'],axis=1)
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)

# %% [markdown]
# Model Building

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# %%
model = LogisticRegression(solver='liblinear',random_state=0)
model.fit(x_train,y_train)

# %%
pred =model.predict(x_test)
pred

# %%
difference = abs(pred-y_test)
difference.mean()

# %% [markdown]
# According to this model, the error is 11.03


