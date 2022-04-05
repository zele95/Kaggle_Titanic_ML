# %%
# In this project, it is necessary to find a model
# that predicts which passengers survived the sinking of the Titanic

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# Load the passenger data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# print(df_train.head())
# print(df_test.head())

print(df_train.info())
print(df_train.describe().transpose())


# print(df_train.Embarked.value_counts(dropna = False))
# print(df_train.Cabin.value_counts(dropna = False))

# %%
# Exploratory Data Analisys

sns.heatmap(df_train.corr(),cmap='OrRd',annot=True,fmt=".2f")
plt.show()

sns.pairplot(df_train,hue = 'Survived')
plt.show()

plt.pie(df_train.Survived.value_counts().values,\
     labels = df_train.Survived.value_counts().index, autopct = '%d%%')
plt.show()

sns.histplot(data = df_train, x = 'Fare', hue = 'Survived', kde = True)
plt.show()

sns.scatterplot(data = df_train, x = 'Age', y = 'Fare', hue = 'Survived')
plt.show()

fig,axes=plt.subplots(1,3,figsize=(12,5))
sns.countplot(x="Sex",hue="Survived",data=df_train,ax=axes[0])
sns.countplot(x="Pclass",hue="Survived",data=df_train,ax=axes[1])
sns.countplot(x="Embarked",hue="Survived",data=df_train,ax=axes[2])
plt.suptitle("Number of Survivals Based On Sex,Pclass and Embarked",fontsize=12,y=1.05)
plt.tight_layout()
plt.show()
#%%
# Data preprocessing

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

# %%
# Create new features

# Whether they were alone on the boat or not
df_train['Alone'] = (df_train['SibSp'] + df_train['SibSp']) == 0
df_train['Alone'] = df_train['Alone'].astype(int)

# With how many family members
df_train['Family'] = df_train['SibSp'] + df_train['SibSp']

# Create different class columns

# print(df_train.Pclass.value_counts())
# print(pd.get_dummies(df_train.Pclass).shape)
df_train[['FirstClass','SecondClass','ThirdClass']] = pd.get_dummies(df_train.Pclass)
# print(df_train.head())
# df_test[['FirstClass','SecondClass','ThirdClass']] = pd.get_dummies(df_test.Pclass)

# df_train['Embarked'] = pd.get_dummies(df_train.Embarked)

# %%
# Select the desired features
features = df_train[['Sex','FirstClass','SecondClass','ThirdClass',\
                    # 'Alone',\
                    'Family',\
                    # 'Parch',\
                    # 'Embarked',\
                    'Age']]

survival = df_train.Survived

# Perform train and test split
X_train, X_test,y_train, y_test = train_test_split(features,survival, test_size = 0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
stdScaler = StandardScaler()
X_train_std = stdScaler.fit_transform(X_train)
X_test_std = stdScaler.transform(X_test)

# %%
# Logistic Regression

# Create and train the model
model = LogisticRegression() 
model.fit(X_train_std,y_train)

# Analyze the coefficients
print('Coefficients:')
print(list(zip(features,model.coef_[0])))

# Score the model on the test data
print(f'Logistic Regression model score: {model.score(X_test_std,y_test)}')

# %%
# K-Neighbors Classifier

# choose the best value of k
accuracies = []
i = 20
for k in range(1,i):
  classifier = KNeighborsClassifier(k)
  classifier.fit(X_train_std,y_train)
  accuracies.append(classifier.score(X_test_std,y_test))
klist = range(1,i)

plt.plot(klist,accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.grid()
plt.show()

k = 10

# fit the model
model2 = KNeighborsClassifier(k)
model2.fit(X_train,y_train)

print(f'K-Neighbors model score: {model2.score(X_test,y_test)}')

# %%
# Naive Bayes Classifier

# create and fit Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

# see the accuracy
print(f'Naive Bayes model score: {classifier.score(X_test,y_test)}')

# %%


