# Logistic Regression on Students' Performance Dataset
This project is about performing logistic regression on a dataset that contains information about students' performance in math, reading, and writing, as well as other personal and demographic information. The goal of this project is to build a model that can predict the mean score of the three subjects based on the given information.

## Data
The dataset used in this project is called "StudentsPerformance.csv". It contains 1000 rows and 8 columns. The columns are as follows:

gender - the gender of the student (male or female)
race/ethnicity - the race/ethnicity of the student (group A, B, C, D, or E)
parental level of education - the highest level of education achieved by the student's parents (some high school, high school, some college, associate's degree, bachelor's degree, or master's degree)
lunch - whether or not the student receives free or reduced lunch (standard or free/reduced)
test preparation course - whether or not the student completed a test preparation course (completed or none)
math score - the score the student received on the math portion of the exam (out of 100)
reading score - the score the student received on the reading portion of the exam (out of 100)
writing score - the score the student received on the writing portion of the exam (out of 100)
### Exploratory Data Analysis
In the beginning, the necessary packages are imported, and the dataset is loaded. The dataset is then examined using data.info() and data.describe(). The mean score column is added to help in identifying overall performance.

Label encoding is used to convert categorical data into numeric data. Afterward, gender and race/ethnicity are analyzed with the help of sns.countplot(). Similarly, test preparation course and lunch are analyzed with the help of plt.pie() and sns.barplot() respectively.

Finally, a pairplot and heatmap are used to visualize the relationship between the columns.

## Data Preprocessing
After analyzing the categorical columns, the relationship between the columns is examined. Data preprocessing is done by dropping math score, reading score, and writing score columns as they are not required for our model.

The dataset is split into training and testing data using train_test_split().

## Model Building
Logistic Regression is used to build a model. The model is trained using the training data and tested using the testing data. The predictions are then compared with the actual values using confusion_matrix() and classification_report(). The error is calculated as the mean absolute difference between the predicted and actual values.

## Conclusion
Based on the analysis and model building, it is clear that students who completed the test preparation course, had a standard lunch, and had parents with above college-level education performed better. The logistic regression model built has an error rate of 11.03.
