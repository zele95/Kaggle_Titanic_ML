# %%
# In this project a Logistic Regression model is created
# that predicts which passengers survived the sinking of the Titanic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the passenger data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# print(df_train.head())
# print(df_test.head())

print(df_train.info())

print(df_train.Embarked.value_counts(dropna = False))
print(df_train.Cabin.value_counts())

# %%
# Exploratory Data Analisys
plt.pie(df_train.Survived.value_counts().values,\
     labels = df_train.Survived.value_counts().index, autopct = '%d%%')
plt.show()

sns.histplot(data = df_train, x = 'Fare', hue = 'Survived', kde = True)
plt.show()

sns.scatterplot(data = df_train, x = 'Age', y = 'Fare', hue = 'Survived')
plt.show()
#%%

# Update sex column to numerical
df_train.Sex.replace(['female','male'],[1,0],inplace = True)
df_train.drop(labels = ['Cabin'], axis = 1)

print(df_train.head())
df_test.Sex.replace(['female','male'],[1,0],inplace = True)
print(df_test.head())

# Fill the nan values in the age column
# print(df_train['Age'].values)
df_train.Age.fillna(df_train.Age.mean(),inplace = True)
# print(df_train['Age'].values)
# print(df_train.Age.describe())


# Create a first class column
# print(df_train.Pclass.value_counts())
# print(pd.get_dummies(df_train.Pclass).shape)
df_train[['FirstClass','SecondClass','ThirdClass']] = pd.get_dummies(df_train.Pclass)
# print(df_train.head())
df_test[['FirstClass','SecondClass','ThirdClass']] = pd.get_dummies(df_test.Pclass)

# Create a second class column


# Select the desired features
features = df_train[['Sex','Age','FirstClass','SecondClass','ThirdClass']]

# features_test = df_test[['Sex','Age','FirstClass','SecondClass']]

survival = df_train.Survived

# Perform train, test, split
X_train, X_test,y_train, y_test=train_test_split(features,survival, test_size = 0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
stdScaler = StandardScaler()
X_train = stdScaler.fit_transform(X_train)
X_test = stdScaler.transform(X_test)

# Create and train the model
model = LogisticRegression() 
model.fit(X_train,y_train)

# Score the model on the train data
print(f'Train model score: {model.score(X_train,y_train)}')

# Score the model on the test data
print(f'Test model score: {model.score(X_test,y_test)}')


# Analyze the coefficients
print('Coefficients:')
print(list(zip(features,model.coef_[0])))

# %%

X_train, X_test,y_train, y_test=train_test_split(features,survival, test_size = 0.2)


accuracies = []
for k in range(1,20):
  classifier = KNeighborsClassifier(k)
  classifier.fit(X_train,y_train)
  accuracies.append(classifier.score(X_test,y_test))
klist = range(1,20)

plt.plot(klist,accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.grid()
plt.show()

k = 10
model2 = KNeighborsClassifier(k)

model2.fit(X_train,y_train)

print(f'Test model score: {model2.score(X_test,y_test)}')

