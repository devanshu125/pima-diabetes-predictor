import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy import stats

df = pd.read_csv('diabetes.csv')

## Cleaning

# make age_category column which will help in filling values which are 0
df['age_category'] = pd.cut(df['Age'], bins=[0, 21, 31, 41, 51, 61, 71, 81, np.inf], labels=[1,2,3,4,5,6,7,8])
df['age_category'] = df['age_category'].astype(int)

# Fill insulin column with respect to age category
agecat = [1,2,3,4,5,6,7,8]

for agec in agecat:
    df['Insulin'].replace(0, round(df[df['age_category'] == agec]['Insulin'].mean(), 0), inplace=True)
    
# Fill rest with same strategy
features = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
for feature in features:
    for agec in agecat:
        if df[feature].dtype == 'float64':
            df[feature].replace(0, round(df[df['age_category'] == agec][feature].mean(), 0), inplace=True)
        else:
            df[feature].replace(0, round(df[df['age_category'] == agec][feature].mean(), 0), inplace=True)

## Outliers

# Z-Score
z = np.abs(stats.zscore(df))
threshold = 3
df = df[(z < 3).all(axis=1)]

# IQR score
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Splitting train and test set
from sklearn.model_selection import train_test_split

X = df.drop(['Outcome', 'age_category', 'DiabetesPedigreeFunction'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Use Random Forest Classifier to predict
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Saving model to disk
pickle.dump(rfc, open('model.pkl', 'wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# test it
print(model.predict([[5, 146, 72, 33, 98, 35, 60]]))

# ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)