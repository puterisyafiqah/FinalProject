
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import warnings 
warnings.filterwarnings("ignore")

df = pd.read_csv(r"healthcare-dataset-stroke-data.csv")

print(df.head())


print(df.shape)

print(df.isna().sum())

print(df.info())

df[df.duplicated()].sum()

fig, axs = plt.subplots(3, figsize = (7,7))
plt1 = sns.boxplot(df['age'], ax = axs[0])
plt4 = sns.boxplot(df['avg_glucose_level'], ax = axs[1])
plt5 = sns.boxplot(df['bmi'], ax = axs[2])
plt.tight_layout()

df.describe()

plt.figure(figsize=(8,6))
sns.countplot(x = 'stroke', data = df)
plt.show()


sns.distplot(df['age'], hist=True, kde=True)


sns.distplot(df['avg_glucose_level'], hist=True, kde=True)


sns.distplot(df['bmi'], hist=True, kde=True)

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)

df['bmi'].fillna(df['bmi'].median(),inplace=True)

df=df.drop('id',axis=1)

df.drop(df[df['gender'] == 'Other'].index, inplace = True)

categorical_data=df.select_dtypes(include=['object']).columns
le=LabelEncoder()
df[categorical_data]=df[categorical_data].apply(le.fit_transform)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote=SMOTE()
X_train,y_train=smote.fit_resample(X_train,y_train)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=25)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)

sns.heatmap(con_mat, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

print(classification_report(y_test, y_pred))
